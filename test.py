import argparse
from typing import Any, Callable, Dict, List

import torch
import onnx
import onnxruntime
from torchsummaryX import summary
from thop import clever_format, profile
from tqdm import tqdm
import numpy as np

import tools
from config import cfg, size_fix
from dataset import augment
from dataset.eval_dataset import EvalDataset
from dataset import SAMPLE_GETTER_REGISTER, RECOVER_BBOXES_REGISTER
from eval.evaluator import Evaluator


# onnxruntime.set_default_logger_severity(0)

def onnx_model_for_benchmark(onnx_path: str, cuda: bool=True):

    def model(x: np.ndarray):
        ort_input = {ort_session.get_inputs()[0].name: x}
        return ort_session.run(None, ort_input)[0]

    load_model = onnx.load(onnx_path)
    onnx.checker.check_model(load_model)
    providers = [] if cuda else ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    return model

def onnx_model_for_eval(onnx_path: str, cuda: bool=True):
    onnx_model = onnx_model_for_benchmark(onnx_path, cuda)

    def model(x: torch.Tensor):
        return torch.from_numpy(onnx_model(x.numpy()))

    return model

def torch_model_for_eval(cfg_path: str, weight_path: str, device: str='cuda', qat=False, quant=False):
    model = tools.build_model(
        cfg_path, weight_path, None, device=device,
        dataparallel=not(quant) and device=='cuda', qat=qat, quantized=quant,
    )[0]
    model.eval()
    return model

def evaluate(config, args):
    if args.onnx:
        model = onnx_model_for_eval(args.onnx, args.device == 'cuda')
    else:
        model = torch_model_for_eval(args.cfg, args.weight, device=args.device)
    eval_dataset = EvalDataset(config)
    evaluator = Evaluator(model, eval_dataset, config)
    AP = evaluator.evaluate()
    tools.print_metric(AP)

def _prepare_images(files: List[str], process: Callable):
    sg = SAMPLE_GETTER_REGISTER['voc'](mode='test', classes=None)
    images = []
    for f in files:
        image, shape = sg(f)
        images.append([process(image, [])[0], torch.FloatTensor(shape)])
    return images

def _print_statistics(statistic: Dict[str, Any]):
    stat = statistic.copy()
    for k in stat:
        if k in {'mean', 'std', '3std', 'min', 'max'}:
            stat[k] /= 1e6
        elif k == 'percent':
            stat[k] *= 100
    if stat.get('percent') is None:
        print('{}:'.format(stat['name']))
    else:
        print('{} ({:.1f}%):'.format(stat['name'], stat['percent']))
    print('\tAverage: {:.2f}±{:.3f} ms'.format(stat['mean'], stat['3std']))
    print('\tRange: {:.2f} ms ~ {:.2f} ms'.format(stat['min'], stat['max']))

def benchmark_onnx(config, args):
    torch.backends.cudnn.benchmark = True

    with open(config.dataset.eval_txt_file, 'r') as fr:
        files = [line.strip() for line in fr.readlines() if len(line.strip()) != 0][:100]

    model = onnx_model_for_benchmark(args.onnx, args.device == 'cuda')
    size = size_fix(args.size)

    # warm up
    print('warmimg up')
    for _ in range(50):
        model(np.random.randn(1, 3, *size).astype(np.float32))

    process = augment.Compose([
        augment.Resize(size),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        augment.HWCtoCHW(),
    ])
    print('loading images')
    images = _prepare_images(files, process)

    total_timer = tools.TicToc('TOTAL')
    forward_timer = tools.TicToc('FORWARD')
    convert_timer = tools.TicToc('CONVERT')
    nms_timer = tools.TicToc('NMS')

    input_shape = torch.FloatTensor(size).to(args.device, non_blocking=False)
    threshold = args.threshold
    nms_iou = args.nms_iou
    for image, shape in tqdm(images):
        image = image[None, ...]
        shape = shape[None, ...].to(args.device)
        total_timer.tic()
        forward_timer.tic()
        pred = model(image)
        forward_timer.toc()
        convert_timer.tic()
        bboxes = RECOVER_BBOXES_REGISTER[args.dataset](
            torch.from_numpy(pred).to(args.device), input_shape, shape
        )[0]
        convert_timer.toc()
        nms_timer.tic()
        tools.torch_nms(bboxes, threshold, nms_iou)
        nms_timer.toc()
        total_timer.toc()

    print('BENCHMARK: {} images in {} size'.format(len(images), size))
    timers = [total_timer, forward_timer, convert_timer, nms_timer]
    stats = [t.statistics() for t in timers]
    for s in stats:
        s['percent'] = s['mean'] / stats[0]['mean']
        _print_statistics(s)

def benchmark(config, args):
    torch.backends.cudnn.benchmark = True

    with open(config.dataset.eval_txt_file, 'r') as fr:
        files = [line.strip() for line in fr.readlines() if len(line.strip()) != 0][:100]

    model = tools.build_model(
        config.model.cfg_path, args.weight, None, device=args.device, dataparallel=False,
        qat=args.qat, quantized=args.quant, backend=args.backend,
    )[0]
    size = size_fix(args.size)
    process = augment.Compose([
        augment.Resize(size),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        augment.ToTensor(args.device),
    ])
    print('loading images')
    images = _prepare_images(files, process)

    total_timer = tools.TicToc('TOTAL')
    forward_timer = tools.TicToc('FORWARD')
    convert_timer = tools.TicToc('CONVERT')
    nms_timer = tools.TicToc('NMS')

    input_shape = torch.FloatTensor(size).to(args.device, non_blocking=False)
    threshold = args.threshold
    nms_iou = args.nms_iou

    # warm up
    print('warmimg up')
    with torch.no_grad():
        for _ in range(5):
            model(torch.rand(1, 3, *size).to(args.device))
            torch.cuda.synchronize()
        for image, shape in tqdm(images):
            image = image.unsqueeze_(0).to(args.device, non_blocking=False)
            shape = shape.unsqueeze_(0).to(args.device, non_blocking=False)
            total_timer.tic()
            forward_timer.tic()
            pred = model(image)
            torch.cuda.synchronize()
            forward_timer.toc()
            convert_timer.tic()
            bboxes = RECOVER_BBOXES_REGISTER[args.dataset](pred, input_shape, shape)[0]
            torch.cuda.synchronize()
            convert_timer.toc()
            nms_timer.tic()
            tools.torch_nms(bboxes, threshold, nms_iou)
            torch.cuda.synchronize()
            nms_timer.toc()
            total_timer.toc()

    print('BENCHMARK: {} images in {} size'.format(len(images), size))
    timers = [total_timer, forward_timer, convert_timer, nms_timer]
    stats = [t.statistics() for t in timers]
    for s in stats:
        s['percent'] = s['mean'] / stats[0]['mean']
        _print_statistics(s)

def model_summary(_, args):
    model = tools.build_model(args.cfg, args.weight, device='cpu', dataparallel=False)[0]
    # print(model)
    inputs = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, inputs=(inputs, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print('MACs: {}, params: {}'.format(flops, params))
    # summary(model, torch.zeros((1, 3, args.size, args.size)))

def time_forward(config, args):
    model = tools.build_model(
        config.model.cfg_path, args.weight, None, device=args.device, dataparallel=False,
        qat=args.qat, quantized=args.quant, backend=args.backend,
    )[0]
    size = args.size
    avg_time = tools.compute_time(model, input_size=(3, size, size), batch_size=args.bs)
    print(f'{avg_time:.2f} ms')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test configuration")
    parser.add_argument(
        'mode', help='test mode', type=str, choices=('eval', 'benchmark', 'summary', 'time')
    )
    parser.add_argument('--yaml', default='yamls/yolo-lite.yaml', required=False)
    parser.add_argument('--cfg', help='model cfg file', required=False)
    parser.add_argument('--weight', help='model weight', required=False)
    parser.add_argument('--onnx', help='onnx file', required=False)
    parser.add_argument('--dataset', help='dataset name', required=False)
    parser.add_argument('--size', help='test image size', type=int, default=512)
    parser.add_argument('--bs', help='batch size', type=int, default=48)
    parser.add_argument('--iou', help='test AP iou', type=int, default=0.5)
    parser.add_argument('--nms-iou', help='NMS iou', type=float, default=0.45)
    parser.add_argument('--threshold', help='test score threshold', type=float, default=0.1)
    parser.add_argument('--device', help='device', type=str, default='cuda')
    parser.add_argument('--qat', help='QAT model', action='store_true', default=False)
    parser.add_argument('--quant', help='quantized model', action='store_true', default=False)
    parser.add_argument('--backend', help='quantized backend', type=str, default='qnnpack')
    args = parser.parse_args()
    if args.yaml:
        cfg.merge_from_file(args.yaml)
        if args.dataset is None:
            args.dataset = cfg.dataset.name.lower()
    if args.cfg:
        cfg.model.cfg_path = args.cfg
    cfg.eval.input_size = args.size
    cfg.eval.score_threshold = args.threshold
    cfg.eval.iou_threshold = args.nms_iou
    cfg.eval.map_iou = args.iou
    cfg.eval.batch_size = args.bs
    cfg.freeze()
    {
        'eval': evaluate,
        'benchmark': benchmark_onnx if args.onnx else benchmark,
        'summary': model_summary,
        'time': time_forward,
    }[args.mode](cfg, args)
