import argparse
from typing import Any, Callable, Dict, List

import torch
import onnx
import onnxruntime
from torchsummaryX import summary
from tqdm import tqdm
import numpy as np

import tools
from config import cfg, size_fix
from dataset import augment
from dataset.eval_dataset import EvalDataset
from dataset.sample import SampleGetter
from eval.evaluator import Evaluator, convert_pred


def onnx_model(onnx_path: str):

    def model(x: np.ndarray):
        ort_input = {ort_session.get_inputs()[0].name: x}
        return ort_session.run(None, ort_input)[0]

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_path)
    return model

def evaluate(config, args):
    model = tools.build_model(
        cfg.model.cfg_path, args.weight, None, device=args.device, dataparallel=True
    )[0]
    eval_dataset = EvalDataset(config)
    evaluator = Evaluator(model, eval_dataset, config)
    model.eval()
    mAP = evaluator.evaluate()
    for kls, acc in mAP.classes.items():
        print('AP@{} = {:.2f}%'.format(kls, acc*100))
    print('mAP = {:.2f}%'.format(mAP.mean*100))

def prepare_images(files: List[str], process: Callable):
    sg = SampleGetter(None, mode='test')
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
    print('{} ({:.1f}%):'.format(stat['name'], stat['percent']))
    print('\tAverage: {:.2f}±{:.3f} ms'.format(stat['mean'], stat['3std']))
    print('\tRange: {:.2f} ms ~ {:.2f} ms'.format(stat['min'], stat['max']))

def benchmark_onnx(config, args):
    torch.backends.cudnn.benchmark = True

    with open(config.dataset.eval_txt_file, 'r') as fr:
        files = [line.strip() for line in fr.readlines() if len(line.strip()) != 0][:100]

    model = onnx_model(args.onnx)
    size = size_fix(args.size)
    process = augment.Compose([
        augment.Resize(size),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        augment.HWCtoCHW(),
    ])
    print('loading images')
    images = prepare_images(files, process)

    total_timer = tools.TicToc('TOTAL')
    forward_timer = tools.TicToc('FORWARD')
    convert_timer = tools.TicToc('CONVERT')
    nms_timer = tools.TicToc('NMS')

    input_shape = torch.FloatTensor(size).to(args.device)
    threshold = args.threshold
    nms_iou = args.nms_iou
    for image, shape in tqdm(images):
        image = image[None, ...]
        shape = shape[None, ...]
        total_timer.tic()
        forward_timer.tic()
        pred = model(image)
        forward_timer.toc()
        convert_timer.tic()
        bboxes = convert_pred(torch.from_numpy(pred).to(args.device), input_shape, shape)[0]
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
        cfg.model.cfg_path, args.weight, None, device=args.device, dataparallel=False,
        qat=args.qat, quantized=args.quant, backend=args.backend,
    )[0]
    size = size_fix(args.size)
    process = augment.Compose([
        augment.Resize(size),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        augment.ToTensor(args.device),
    ])
    print('loading images')
    images = prepare_images(files, process)

    # warm up
    print('warmimg up')
    for _ in range(5):
        with torch.no_grad():
            model(torch.rand(1, 3, *size).to(args.device))

    total_timer = tools.TicToc('TOTAL')
    forward_timer = tools.TicToc('FORWARD')
    convert_timer = tools.TicToc('CONVERT')
    nms_timer = tools.TicToc('NMS')

    with torch.no_grad():
        input_shape = torch.FloatTensor(size).to(args.device)
        threshold = args.threshold
        nms_iou = args.nms_iou
        for image, shape in tqdm(images):
            image = image.unsqueeze_(0).to(args.device)
            shape = shape.unsqueeze_(0).to(args.device)
            total_timer.tic()
            forward_timer.tic()
            pred = model(image)
            forward_timer.toc()
            convert_timer.tic()
            bboxes = convert_pred(pred, input_shape, shape)[0]
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

def model_summary(config, args):
    model = tools.build_model(args.cfg, device='cpu', dataparallel=False)[0]
    summary(model, torch.zeros((1, 3, args.size, args.size)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test configuration")
    parser.add_argument('mode', help='test mode', type=str, choices=('eval', 'benchmark', 'summary'))
    parser.add_argument('--cfg', help='model cfg file', required=False)
    parser.add_argument('--weight', help='model weight', required=False)
    parser.add_argument('--onnx', help='onnx file', required=False)
    parser.add_argument('--size', help='test image size', type=int, default=512)
    parser.add_argument('--iou', help='test AP iou', type=int, default=0.5)
    parser.add_argument('--nms-iou', help='NMS iou', type=int, default=0.45)
    parser.add_argument('--threshold', help='test score threshold', type=float, default=0.1)
    parser.add_argument('--bs', help='batch size', type=int, default=16)
    parser.add_argument('--device', help='device', type=str, default='cuda')
    parser.add_argument('--qat', help='QAT model', action='store_true', default=False)
    parser.add_argument('--quant', help='quantized model', action='store_true', default=False)
    parser.add_argument('--backend', help='quantized backend', type=str, default='fbgemm')
    args = parser.parse_args()
    cfg.model.cfg_path = args.cfg
    cfg.eval.input_size = args.size
    cfg.eval.score_threshold = args.threshold
    cfg.eval.iou_threshold = args.nms_iou
    cfg.eval.map_iou = args.iou
    cfg.dataset.eval_txt_file = cfg.dataset.eval_txt_file
    cfg.eval.batch_size = args.bs
    cfg.freeze()
    {
        'eval': evaluate,
        'benchmark': benchmark,
        'summary': model_summary,
    }[args.mode](cfg, args)
