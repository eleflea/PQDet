from xml.etree.ElementTree import parse

import numpy as np
import torch
from yacs.config import CfgNode as CN

from dataset import augment
from dataset.base_sample import BaseSampleGetter, recover_bboxes_prediction


class VOCSampleGetter(BaseSampleGetter):
    '''VOC dataset
    get sample by image path

    mode: in 'train', 'eval' or 'test'
    train mode should return bboxes
    eval mode should return images, filename, shapes, bboxes, diffs
    test mode should return images, shapes

    bboxes is (xmin, ymin, xmax, ymax, class_index) in absolute coordinate
    diffs is 1-D 01 array, label whose diff==1 wont participate in training
    and will be ignored in eval
    '''
    def label(self, img_path: str):
        '''this method returns bboxes in train mode, bboxes and diffs otherwise
        '''
        label_path = img_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
        root = parse(label_path).getroot()
        bbs, diffs = [], []
        obj_tags = root.findall('object')
        for t in obj_tags:
            diff = int(t.find('difficult').text)
            if self.is_train and diff == 1:
                continue
            cls_name = t.find('name').text
            cls_idx = self.cls_to_idx[cls_name]
            box_tag = t.find('bndbox')
            x1 = box_tag.find('xmin').text
            y1 = box_tag.find('ymin').text
            x2 = box_tag.find('xmax').text
            y2 = box_tag.find('ymax').text
            box = [float(x1), float(y1), float(x2), float(y2), cls_idx]
            bbs.append(box)
            diffs.append(diff)
        bbs = np.array(bbs, dtype=np.float32)
        if self.is_train:
            return bbs
        return bbs, np.array(diffs)

    def set_train_augment(self, augment_cfg: CN, input_size, img_path_sampler):
        self.train_augment = augment.Compose([
            augment.RandomHFlip(p=augment_cfg.hflip_p),
            augment.RandomVFlip(p=augment_cfg.vflip_p),
            augment.RandomSafeCrop(p=augment_cfg.crop_p),
            augment.ColorJitter(
                brightness=[-0.1, 0.1],
                contrast=[0.8, 1.2],
                saturation=[0.1, 2],
                p=augment_cfg.color_p,
            ),
            augment.Resize(input_size),
            augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        sampler = lambda: super(VOCSampleGetter, self).train(img_path_sampler())
        self.compose_augment = augment.Compose([
            augment.Mixup(sampler, p=augment_cfg.mixup_p, beta=1.5),
            augment.ToTensor('cpu'),
        ])
        # aug2 = augment.Compose([
        #     augment.Mosaic(sampler, p=1, size=input_size),
        #     augment.Mixup(sampler, p=0, beta=1.5),
        #     augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     augment.ToTensor('cpu'),
        # ])
        return self

    def set_eval_augment(self, input_size):
        self.eval_augment = eval_augment_voc(input_size, 'cpu')
        return self

    def train(self, img_path: str):
        image, bboxes = super(VOCSampleGetter, self).train(img_path)
        return self.compose_augment(image, bboxes)

def eval_augment_voc(input_size, device):
    return augment.Compose([
        augment.Resize(input_size),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        augment.ToTensor(device),
    ])

def _voc_affine_bboxes(input_size: torch.Tensor, batch_original_size: torch.Tensor):
    resize_ratio, _ = (input_size / batch_original_size).min(dim=-1) # (B, )
    delta = (input_size - (resize_ratio.unsqueeze_(-1) * batch_original_size).round()) / 2 # (B, 2)
    return delta.floor(), resize_ratio.unsqueeze_(-1)

def recover_bboxes_prediction_voc(
    batch_pred_bbox: torch.Tensor,
    input_size: torch.Tensor,
    batch_original_size: torch.Tensor,
    ) -> torch.Tensor:
    return recover_bboxes_prediction(
        batch_pred_bbox, input_size, batch_original_size, _voc_affine_bboxes
    )