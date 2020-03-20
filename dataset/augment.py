import math
from collections import namedtuple
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

def rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    r, g, b = img.unbind(dim=0)

    maxc = img.max(0)[0]
    minc = img.min(0)[0]

    deltac = maxc - minc
    s = deltac / maxc
    s[torch.isnan(s)] = 0.

    # avoid division by zero
    deltac = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac)

    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac

    maxg = g == maxc
    maxr = r == maxc

    h = (4.0 + gc - rc)
    h[maxg] = 2.0 + rc[maxg] - bc[maxg]
    h[maxr] = bc[maxr] - gc[maxr]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0

    h = 2 * math.pi * h
    return torch.stack([h, s, maxc], dim=0)

def hsv_to_rgb(img: torch.Tensor) -> torch.Tensor:
    h, s, v = img.unbind(dim=0)
    h.div_(2 * math.pi)

    hi = torch.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    # pylint: disable-msg=not-callable
    one = torch.tensor(1.).to(img.device)
    p = v * (one - s)
    q = v * (one - f * s)
    t = v * (one - (one - f) * s)

    out = torch.stack([hi, hi, hi], dim=0)

    out[out == 0] = torch.stack((v, t, p), dim=0)[out == 0]
    out[out == 1] = torch.stack((q, v, p), dim=0)[out == 1]
    out[out == 2] = torch.stack((p, v, t), dim=0)[out == 2]
    out[out == 3] = torch.stack((p, q, v), dim=0)[out == 3]
    out[out == 4] = torch.stack((t, p, v), dim=0)[out == 4]
    out[out == 5] = torch.stack((v, p, q), dim=0)[out == 5]
    return out

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

class Resize:

    def __init__(self, size: Union[_size_T, Callable[[], _size_T]], pad_val: int=128):
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

class ToTensor:

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, img: np.ndarray, bboxes: _bboxes_T):
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img).to(self.device)
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
