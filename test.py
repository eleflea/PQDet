import argparse
from typing import Callable, List, Dict, Any

import torch
from tqdm import tqdm

import tools
from config import cfg, size_fix
from dataset import augment
from dataset.eval_dataset import EvalDataset
from dataset.sample import SampleGetter
from eval.evaluator import Evaluator, convert_pred
from model.newyolo import YOLOv3


def trained_model(cfg_path: str, weight_path: str, device):
    model = torch.nn.DataParallel(YOLOv3(cfg_path)).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    return model

def evaluate(config, args):
    model = trained_model(cfg.model.cfg_path, args.weight, args.device)
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

def benchmark(config, args):
    torch.backends.cudnn.benchmark = True

    with open(config.dataset.eval_txt_file, 'r') as fr:
        files = [line.strip() for line in fr.readlines() if len(line.strip()) != 0][:100]

    model = trained_model(cfg.model.cfg_path, args.weight, args.device).module
    size = size_fix(args.size)
    process = augment.Compose([
        augment.Resize(size),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        augment.ToTensor('cpu'),
    ])
    images = prepare_images(files, process)

    # warm up
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test configuration")
    parser.add_argument('mode', help='test mode', type=str, choices=('eval', 'benchmark'))
    parser.add_argument('--weight', help='model weight')
    parser.add_argument('--cfg', help='model cfg file')
    parser.add_argument('--size', help='test image size', type=int, default=512)
    parser.add_argument('--iou', help='test AP iou', type=int, default=0.5)
    parser.add_argument('--nms-iou', help='NMS iou', type=int, default=0.45)
    parser.add_argument('--threshold', help='test score threshold', type=float, default=0.1)
    parser.add_argument('--bs', help='batch size', type=int, default=16)
    parser.add_argument('--device', help='device', type=str, default='cuda')
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
    }[args.mode](cfg, args)
