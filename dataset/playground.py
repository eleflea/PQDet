from math import pi
from time import sleep, time_ns
from typing import Sequence

import cv2
import numpy as np
import torch

from config import cfg
from dataset import augment
from dataset.sample import VOCXmlParser

CLS_NAMES = cfg.dataset.classes
CLS_COLOR_PLATE = np.random.randint(0, 256, size=(len(CLS_NAMES), 3))

def read_sample(img_path: str):
    img = cv2.imread(img_path)
    parser = VOCXmlParser(CLS_NAMES, ['bbox'])
    bboxes = parser(img_path.replace('jpg', 'xml'))[0]
    return img, bboxes

def process_sample(img, bboxes: np.ndarray):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    process = augment.Compose([
        augment.ToTensor(device),
        augment.RandomHFlip(p=0.5),
        augment.RandomCrop(p=0.75),
        augment.CutOut(p=0.8, size=30, n_holes=3),
        augment.ColorJitter(
            brightness=[-0.1, 0.1],
            contrast=[0.6, 1.4],
            saturation=[0.1, 2],
            hue=[-pi/3, pi/3],
        ),
        augment.Resize(lambda: [512, 512]),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img, bboxes = process(img, bboxes)
    return img, bboxes

def deconvert(img):
    dn = augment.DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = (dn(img, [])[0]*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def draw_box(img, bboxes):
    for box in bboxes:
        color = [int(i) for i in CLS_COLOR_PLATE[int(box[4])]]
        x1, y1, x2, y2 = [int(n) for n in box[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if len(box) == 6:
            text = '{} {:.3f}'.format(CLS_NAMES[int(box[4])], box[-1])
        else:
            text = '{}'.format(CLS_NAMES[int(box[4])])
        cv2.putText(img, text, (x1, y1-5), 0, 0.4, color)
    return img

def continuous_show(img_paths: Sequence[str], inter: float=1):

    def sampler():
        img_path = np.random.choice(img_paths)
        img, bboxes = read_sample(img_path)
        processed_img, processed_bboxes = process_sample(img, bboxes.copy())
        return img, bboxes, processed_img, processed_bboxes

    mixup = augment.Mixup(lambda: sampler()[2:], p=0.5, beta=1.5)

    while 1:
        t = time_ns()
        img, bboxes, processed_img, processed_bboxes = sampler()
        processed_img, processed_bboxes = mixup(processed_img, processed_bboxes)
        print('{:.2f}ms'.format((time_ns() - t)/1e6))
        cv2.imshow('original', draw_box(img, bboxes))
        cv2.imshow('augmented', draw_box(deconvert(processed_img), processed_bboxes))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        sleep(inter)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    paths = ['data/images/0000%d.jpg' % i for i in range(32, 36)]
    continuous_show(paths, inter=3)
