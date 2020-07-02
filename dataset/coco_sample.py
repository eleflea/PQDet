import numpy as np
import torch
from yacs.config import CfgNode as CN

from dataset import augment
from dataset.base_sample import BaseSampleGetter, recover_bboxes_prediction


class COCOSampleGetter(BaseSampleGetter):
    '''COCO dataset
    get sample by image path

    use darknet labels (class, xc, yc, w, h) in relative

    mode: in 'train', 'eval' or 'test'
    '''

    def label(self, img_path: str):
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        bbs, diffs = [], []
        fr = open(label_path, 'r')
        for line in fr.readlines():
            ann = line.split(' ')
            # diff always be 0
            diff = 0
            cls_idx = int(ann[0])
            # note here we return a relative bboxes
            # we turn it to absolutely in `self._train` and `self.eval`
            half_rw, half_rh = float(ann[3]) / 2, float(ann[4]) / 2
            rx1 = float(ann[1]) - half_rw
            ry1 = float(ann[2]) - half_rh
            rx2 = float(ann[1]) + half_rw
            ry2 = float(ann[2]) + half_rh
            box = [rx1, ry1, rx2, ry2, cls_idx]
            bbs.append(box)
            diffs.append(diff)
        fr.close()
        bbs = np.array(bbs, dtype=np.float32)
        if self.is_train:
            return bbs
        return bbs, np.array(diffs)

    @staticmethod
    def _relative_to_absolute(bboxes: np.ndarray, shape: np.ndarray):
        bboxes[:, :-1] *= np.tile(shape[[1, 0]], 2)
        return bboxes

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
        sampler = lambda: self._train(img_path_sampler())
        self.compose_augment = augment.Compose([
            augment.Mixup(sampler, p=augment_cfg.mixup_p, beta=1.5),
            augment.ToTensor('cpu'),
        ])
        return self

    def set_eval_augment(self, input_size):
        self.eval_augment = eval_augment_coco(input_size, 'cpu')
        return self

    def _train(self, img_path: str):
        image = self.image(img_path)
        bboxes = self._relative_to_absolute(self.label(img_path), self.shape(image))
        return self.train_augment(image, bboxes)

    def train(self, img_path: str):
        image, bboxes = self._train(img_path)
        return self.compose_augment(image, bboxes)

    def eval(self, img_path: str):
        image = self.image(img_path)
        shape = self.shape(image)
        bboxes, diffs = self.label(img_path)
        bboxes = self._relative_to_absolute(bboxes, shape)
        image = self.eval_augment(image, [])[0]
        return (image, self.file_name(img_path), shape, bboxes, diffs)

def eval_augment_coco(input_size, device):
    return augment.Compose([
        augment.Resize(input_size),
        augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        augment.ToTensor(device),
    ])

def _coco_affine_bboxes(input_size: torch.Tensor, batch_original_size: torch.Tensor):
    resize_ratio, _ = (input_size / batch_original_size).min(dim=-1) # (B, )
    delta = (input_size - (resize_ratio.unsqueeze_(-1) * batch_original_size).round()) / 2 # (B, 2)
    return delta.floor(), resize_ratio.unsqueeze_(-1)

def recover_bboxes_prediction_coco(
    batch_pred_bbox: torch.Tensor,
    input_size: torch.Tensor,
    batch_original_size: torch.Tensor,
    ) -> torch.Tensor:
    return recover_bboxes_prediction(
        batch_pred_bbox, input_size, batch_original_size, _coco_affine_bboxes
    )
