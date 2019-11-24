import argparse

import numpy as np
import torch

import config as cfg
from eval.evaluate import Evaluator
from model.yolov3 import YOLOv3


def main(args):
    model = YOLOv3().cuda()
    state_dict = torch.load(args.weight)
    model.load_state_dict(state_dict['model'])

    evaluator = Evaluator(model)
    model.eval()
    APs = evaluator.mAP()
    for cls in APs:
        AP_mess = 'AP for %s = %.4f\n' % (cls, APs[cls])
        print(AP_mess.strip())
    mAP = np.mean([APs[cls] for cls in APs])
    mAP_mess = 'mAP = %.4f\n' % mAP
    print(mAP_mess.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test configuration")
    parser.add_argument('--weight', help='model weight')

    parser.add_argument('--size', help='test image size', type=int, default=512)
    parser.add_argument('--iou', help='test AP iou', type=int, default=0.5)
    parser.add_argument('--threshold', help='test score threshold', type=float, default=0.01)
    args = parser.parse_args()
    cfg.TEST_INPUT_SIZE = args.size
    cfg.SCORE_THRESHOLD = args.threshold
    cfg.IOU_THRESHOLD = args.iou
    main(args)
