from time import time_ns
from typing import Sequence

import cv2
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from config import cfg
from dataset import augment
from dataset.sample import VOCXmlParser

CLS_NAMES = cfg.dataset.classes
CLS_COLOR_PLATE = np.random.rand(len(CLS_NAMES), 3)

def read_sample(img_path: str):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    parser = VOCXmlParser(CLS_NAMES, train=True)
    bboxes = parser(img_path.replace('jpg', 'xml'))
    return img, bboxes

def process_sample(img, bboxes: np.ndarray):
    process = augment.Compose([
        augment.RandomHFlip(p=0.5),
        augment.RandomSafeCrop(p=0.75),
        augment.CutOut(p=0.8, size=30, n_holes=3),
        augment.ColorJitter(
            brightness=[-0.1, 0.1],
            contrast=[0.8, 1.2],
            saturation=[0.1, 2],
        ),
        augment.Resize(lambda: [512, 512]),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img, bboxes = process(img, bboxes)
    return img, bboxes

def deconvert(img):
    dn = augment.DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = dn(img, [])[0]
    return img

def draw_box(ax, bboxes):
    anns = []
    for box in bboxes:
        color = CLS_COLOR_PLATE[int(box[4])]
        x1, y1, x2, y2 = box[:4]
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        if len(box) == 6:
            text = '{} {:.3f}'.format(CLS_NAMES[int(box[4])], box[-1])
        else:
            text = '{}'.format(CLS_NAMES[int(box[4])])
        anns.append(ax.annotate(text, (x1, y1), color=color, weight='bold', 
            fontsize=14, ha='left', va='bottom'))
    return anns

def continuous_show(img_paths: Sequence[str]):

    def sampler():
        img_path = np.random.choice(img_paths)
        img, bboxes = read_sample(img_path)
        processed_img, processed_bboxes = process_sample(img, bboxes.copy())
        return img, bboxes, processed_img, processed_bboxes

    mixup = augment.Mixup(lambda: sampler()[2:], p=0.5, beta=1.5)

    def reset_patches(ax):
        [p.remove() for p in reversed(ax.patches)]

    def show_img(fig, texts):
        [t.remove() for t in texts]
        texts.clear()
        t = time_ns()
        img, bboxes, processed_img, processed_bboxes = sampler()
        processed_img, processed_bboxes = mixup(processed_img, processed_bboxes)
        print('{:.2f}ms'.format((time_ns() - t)/1e6))
        ax1.imshow(img)
        reset_patches(ax1)
        texts += draw_box(ax1, bboxes)
        ax2.imshow(deconvert(processed_img))
        reset_patches(ax2)
        texts += draw_box(ax2, processed_bboxes)
        fig.canvas.draw()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    texts = []
    ax1.set_title('original', fontsize=16)
    ax1.axis('off')
    ax2.set_title('augmented', fontsize=16)
    ax2.axis('off')
    fig.subplots_adjust(top = 1, bottom = 0.1, right = 1, left = 0, 
                hspace = 0, wspace = 0.05)
    axnext = plt.axes([0.85, 0.02, 0.1, 0.06])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(lambda e: show_img(fig, texts))
    show_img(fig, texts)
    plt.show()

    plt.close(fig)

if __name__ == "__main__":
    paths = ['data/images/0000%d.jpg' % i for i in range(32, 36)]
    continuous_show(paths)
