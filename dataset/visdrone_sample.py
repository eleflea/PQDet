import numpy as np
import torch
from yacs.config import CfgNode as CN

from dataset import augment
from dataset.base_sample import BaseSampleGetter, recover_bboxes_prediction


class VisDroneSampleGetter(BaseSampleGetter):
    '''VisDrone2019 dataset
    get sample by image path

    (x, y, w, h, score, object_categry, truncation, occlusion)
    see https://github.com/VisDrone/VisDrone2018-DET-toolkit for more info.

    mode: in 'train', 'eval' or 'test'
    '''

    def label(self, img_path: str):
        label_path = img_path.replace('images', 'annotations').replace('.jpg', '.txt')
        bbs, diffs = [], []
        fr = open(label_path, 'r')
        for line in fr.readlines():
            ann = line.split(',')
            # omit the ignored regions(0) and others(11) classes
            if int(ann[5]) in {0, 11}:
                continue
            # turn score to difficulty, score == 0 will be ignored in eval
            diff = 0 if int(ann[4]) == 1 else 1
            if self.is_train and diff == 1:
                continue
            # minus 1 because 0 is ignored regions
            cls_idx = int(ann[5]) - 1
            x1 = int(ann[0])
            y1 = int(ann[1])
            x2 = int(ann[0]) + int(ann[2])
            y2 = int(ann[1]) + int(ann[3])
            box = [float(x1), float(y1), float(x2), float(y2), cls_idx]
            bbs.append(box)
            diffs.append(diff)
        fr.close()
        bbs = np.array(bbs, dtype=np.float32)
        if self.is_train:
            return bbs
        return bbs, np.array(diffs)

    def set_train_augment(self, augment_cfg: CN, input_size, img_path_sampler):
        self.train_augment = augment.Compose([
            augment.RandomCrop((416, 416), p=1),
            augment.RandomHFlip(p=augment_cfg.hflip_p),
            augment.RandomVFlip(p=augment_cfg.vflip_p),
            augment.ColorJitter(
                brightness=[-0.1, 0.1],
                contrast=[0.8, 1.2],
                saturation=[0.1, 2],
                p=augment_cfg.color_p,
            ),
            augment.Resize(input_size),
            augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        sampler = lambda: super(VisDroneSampleGetter, self).train(img_path_sampler())
        self.compose_augment = augment.Compose([
            augment.Mixup(sampler, p=augment_cfg.mixup_p, beta=1.5),
            augment.ToTensor('cpu'),
        ])
        return self

    def set_eval_augment(self, _):
        self.eval_augment = eval_augment_visdrone(_, 'cpu')
        return self

    def train(self, img_path: str):
        image, bboxes = super(VisDroneSampleGetter, self).train(img_path)
        return self.compose_augment(image, bboxes)

def eval_augment_visdrone(_, device):
    return augment.Compose([
        augment.ResizeRatio(1.25),
        augment.PadNearestDivisor(),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        augment.ToTensor(device),
    ])

def _visdrone_affine_bboxes(input_size: torch.Tensor, batch_original_size: torch.Tensor):
    resize_ratio = 1.25
    input_size = torch.ceil(resize_ratio * batch_original_size / 32) * 32
    delta = (input_size - resize_ratio * batch_original_size) / 2 # (B, 2)
    return delta.floor(), resize_ratio

def recover_bboxes_prediction_visdrone(
    batch_pred_bbox: torch.Tensor,
    input_size: torch.Tensor,
    batch_original_size: torch.Tensor,
    ) -> torch.Tensor:
    return recover_bboxes_prediction(
        batch_pred_bbox, input_size, batch_original_size, _visdrone_affine_bboxes
    )
