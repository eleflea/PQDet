import math
from collections import namedtuple
from itertools import chain
from typing import Callable, List, Sequence, Tuple, Union, Any

import cv2
import numpy as np
import torch
from numpy import random


_size_T = Union[List[int], Tuple[int, int]]
_triplet_T = Union[List[float], Tuple[float, float, float]]
_range_T = Union[List[float], Tuple[float, float]]
_transform_T = Callable[[Any, np.ndarray], Tuple[np.ndarray, np.ndarray]]
_sampler_T = Callable[[], Tuple[np.ndarray, np.ndarray]]
_bboxes_T = Union[List, np.ndarray]

class RandomCrop:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        h, w= img.shape[:2]
        # 得到可以包含所有bbox的最小bbox
        if len(bboxes) > 0:
            max_bbox = np.concatenate([
                np.min(bboxes[:, 0:2], axis=0),
                np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        else:
            cx, cy = w // 2, h // 2
            max_bbox = np.array([cx, cy, cx+1, cy+1])
        crop_xmin = random.randint(0, max_bbox[0]+1)
        crop_ymin = random.randint(0, max_bbox[1]+1)
        crop_xmax = random.randint(max_bbox[2], w+1)
        crop_ymax = random.randint(max_bbox[3], h+1)

        img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]

        if len(bboxes) != 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return img, bboxes

class RandomHFlip:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        w = img.shape[1]
        img = img[:, ::-1, :]

        if len(bboxes) != 0:
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return img, bboxes

Operation = namedtuple('Operation', ['func', 'params_range'])

class ColorJitter:

    def __init__(self, brightness: _range_T, contrast: _range_T,
        saturation: _range_T, p=1.):
        self.operations = [
            Operation(self.adjust_brightness, brightness),
            Operation(self.adjust_contrast, contrast),
            Operation(self.adjust_saturation, saturation),
        ]
        self.p = p

    @staticmethod
    def adjust_brightness(img: np.ndarray, brightness: _range_T):
        b = random.uniform(*brightness)*255
        return np.clip(img+round(b), 0, 255)

    @staticmethod
    def adjust_contrast(img: np.ndarray, contrast: _range_T):
        c = random.uniform(*contrast)
        return np.clip(img*c, 0, 255).astype(np.int32)

    @staticmethod
    def adjust_saturation(img: np.ndarray, saturation: _range_T):
        img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        s = random.uniform(*saturation)
        img = s * img + (1-s) * img_gray[..., None]
        img = np.clip(img, 0, 255).astype(np.int32)
        return img

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        ops = self.operations.copy()
        random.shuffle(ops)
        img = img.astype(np.int32)
        for op in ops:
            img = op.func(img, op.params_range)
        return img.astype(np.uint8), bboxes

class CutOut:

    def __init__(self, size: int, n_holes: int, p: float=0.5, pad_val: int=128):
        self.p = p
        self.size = size // 2
        self.n_holes = n_holes
        self.pad_val = pad_val

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        h, w = img.shape[:2]
        for _ in range(self.n_holes):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)

            y1 = np.clip(y - self.size, 0, h)
            y2 = np.clip(y + self.size, 0, h)
            x1 = np.clip(x - self.size, 0, w)
            x2 = np.clip(x + self.size, 0, w)

            img[y1:y2, x1:x2, :] = self.pad_val
        return img, bboxes

class Normalize:

    def __init__(self, mean: _triplet_T=(0., 0., 0.), std: _triplet_T=(1., 1., 1.)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        img = img.astype(np.float32, copy=False)
        img = (img/255. - self.mean) / self.std
        return img, bboxes

class DeNormalize:

    def __init__(self, mean: _triplet_T=(0., 0., 0.), std: _triplet_T=(1., 1., 1.)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        np.clip((img * self.std + self.mean)*255., 0, 255, out=img)
        return img.astype(np.uint8), bboxes

_aware_size_T = Union[_size_T, Callable[[], _size_T]]

class Resize:

    def __init__(self, size: _aware_size_T, pad_val: int=128):
        self.pad_val = pad_val
        self.size = size

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if callable(self.size):
            target_h, target_w = self.size()
        else:
            target_h, target_w = self.size
        img_h, img_w = img.shape[:2]

        resize_ratio = min(target_w / img_w, target_h / img_h)
        resize_w = round(resize_ratio * img_w)
        resize_h = round(resize_ratio * img_h)
        image_resized = cv2.resize(img, dsize=(resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        dl = (target_w - resize_w) // 2
        dr = target_w - resize_w - dl
        du = (target_h - resize_h) // 2
        dd = target_h - resize_h - du
        img_padded = np.pad(
            image_resized, ((du, dd), (dl, dr), (0, 0)),
            'constant', constant_values=self.pad_val
        )

        if len(bboxes) != 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dl
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + du
        return img_padded, bboxes

class Mixup:

    def __init__(self, sampler: _sampler_T, p=0.5, beta: float=1.):
        self.sampler = sampler
        self.p = p
        self.beta = beta

    @staticmethod
    def mixup_bboxes(bboxes: _bboxes_T, mixup_factor: float):
        if len(bboxes) == 0:
            return bboxes
        mfs = np.full((len(bboxes), 1), mixup_factor, dtype=np.float32)
        return np.concatenate([bboxes, mfs], axis=-1)

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if random.random() > self.p:
            bboxes = self.mixup_bboxes(bboxes, 1.0)
            return img, bboxes
        img_mix, bboxes_mix = self.sampler()
        lam = random.beta(self.beta, self.beta)
        img = lam * img + (1 - lam) * img_mix
        bboxes = self.mixup_bboxes(bboxes, lam)
        bboxes_mix = self.mixup_bboxes(bboxes_mix, 1-lam)
        bboxes_no_empty = [b for b in [bboxes, bboxes_mix] if len(b) != 0]
        bboxes = np.concatenate(bboxes_no_empty)
        return img, bboxes

class Mosaic:

    def __init__(self, sampler: _sampler_T, size: _aware_size_T, pad_val: int=128, p: float=1):
        self.sampler = sampler
        self.size = size
        self.pad_val = pad_val
        self.p = p

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        if callable(self.size):
            input_h, input_w = self.size()
        else:
            input_h, input_w = self.size
        xc = int(random.uniform(input_w * 0.5, input_w * 1.5))
        yc = int(random.uniform(input_h * 0.5, input_h * 1.5))

        bboxes4 = []
        img4 = np.zeros((input_h * 2, input_w * 2, 3), dtype=np.uint8) + self.pad_val
        other_imgs, other_bboxes = list(zip(*[self.sampler() for _ in range(3)]))
        all_bboxes_copy = np.concatenate([bboxes] + list(other_bboxes), axis=0)
        all_imgs_bboxes = zip(chain([img], other_imgs), chain([bboxes], other_bboxes))
        for i, (image, bboxes) in enumerate(all_imgs_bboxes):
            h, w = image.shape[:2]

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            bboxes[:, [0, 2]] += x1a - x1b
            bboxes[:, [1, 3]] += y1a - y1b
            bboxes4.append(bboxes)

        bboxes4 = np.concatenate(bboxes4, axis=0)
        bboxes4[:, [0, 2]] = np.clip(bboxes4[:, [0, 2]] - input_w / 2, 0, input_w)
        bboxes4[:, [1, 3]] = np.clip(bboxes4[:, [1, 3]] - input_h / 2, 0, input_h)

        img4 = img4[input_h // 2: input_h // 2 + input_h, input_w // 2: input_w // 2 + input_w]

        w = bboxes4[:, 2] - bboxes4[:, 0]
        h = bboxes4[:, 3] - bboxes4[:, 1]
        area = w * h
        area0 = (all_bboxes_copy[:, 2] - all_bboxes_copy[:, 0]) * (all_bboxes_copy[:, 3] - all_bboxes_copy[:, 1])
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        i = (w > 8) & (h > 8) & (area / (area0 + 1e-16) > 0.25) & (ar < 10)
        bboxes4 = bboxes4[i]

        return img4, bboxes4

class ToTensor:

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img).to(self.device)
        return img, bboxes

class HWCtoCHW:

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return img, bboxes

class Empty:

    def __call__(self, *args):
        return args

class Compose:

    def __init__(self, transforms: Sequence[_transform_T]):
        self.transforms = transforms

    def __call__(self, img, bboxes: _bboxes_T):
        for transform in self.transforms:
            img, bboxes = transform(img, bboxes)
        return img, bboxes
