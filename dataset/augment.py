import math
from collections import namedtuple
from typing import Callable, List, Sequence, Tuple, Union, Any

import cv2
import numpy as np
import torch
# from kornia.augmentation.functional
from numpy import random
from torch.nn import functional as F

_size_T = Union[List[int], Tuple[int, int]]
_triplet_T = Union[List[float], Tuple[float, float, float]]
_range_T = Union[List[float], Tuple[float, float]]
_transform_T = Callable[[Any, np.ndarray], Tuple[torch.Tensor, np.ndarray]]
_sampler_T = Callable[[], Tuple[torch.Tensor, np.ndarray]]
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
    
    def __call__(self, img: torch.Tensor, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        _, h, w= img.shape
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

        img = img[:, crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        if len(bboxes) != 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return img, bboxes

class RandomHFlip:

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img: torch.Tensor, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        w = img.shape[2]
        img = img.flip([2])

        if len(bboxes) != 0:
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return img, bboxes

Operation = namedtuple('Operation', ['func', 'params_range', 'color_space'])

class ColorJitter:

    def __init__(self, brightness: _range_T, contrast: _range_T,
        saturation: _range_T, hue: _range_T):
        self.operations = [
            Operation(self.adjust_brightness, brightness, 'rgb'),
            Operation(self.adjust_contrast, contrast, 'rgb'),
            Operation(self.adjust_saturation, saturation, 'hsv'),
            Operation(self.adjust_hue, hue, 'hsv'),
        ]
        self.conversions = {'rgbhsv': rgb_to_hsv, 'hsvrgb': hsv_to_rgb}

    @staticmethod
    def adjust_brightness(img: torch.Tensor, brightness: float):
        return img.add_(brightness).clamp_(0, 1)

    @staticmethod
    def adjust_contrast(img: torch.Tensor, contrast: float):
        return img.mul_(contrast).clamp_(0, 1)

    @staticmethod
    def adjust_saturation(img: torch.Tensor, saturation: float):
        img[1, ...].mul_(saturation).clamp_(0, 1)
        return img

    @staticmethod
    def adjust_hue(img: torch.Tensor, hue: float):
        img[0, ...].add_(hue).fmod_(2 * math.pi)
        return img

    @staticmethod
    def uniform_gen(low: float=0.0, high: float=1.0):
        return random.uniform(low, high)

    def __call__(self, img: torch.Tensor, bboxes: _bboxes_T):
        color_space = 'rgb'
        ops = self.operations.copy()
        random.shuffle(ops)
        for op in ops:
            if color_space != op.color_space:
                img = self.conversions[color_space+op.color_space](img)
            img = op.func(img, self.uniform_gen(*op.params_range))
            color_space = op.color_space
        if color_space != 'rgb':
            img = self.conversions[color_space+'rgb'](img)
        return img, bboxes

class CutOut:

    def __init__(self, size: int, n_holes: int, p: float=0.5, pad_val: float=0.5):
        self.p = p
        self.size = size // 2
        self.n_holes = n_holes
        self.pad_val = pad_val

    def __call__(self, img: torch.Tensor, bboxes: _bboxes_T):
        if random.random() > self.p:
            return img, bboxes
        _, h, w = img.shape
        for _ in range(self.n_holes):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)

            y1 = np.clip(y - self.size, 0, h)
            y2 = np.clip(y + self.size, 0, h)
            x1 = np.clip(x - self.size, 0, w)
            x2 = np.clip(x + self.size, 0, w)

            img[:, y1:y2, x1:x2] = self.pad_val
        return img, bboxes

class Normalize:

    def __init__(self, mean: _triplet_T=(0., 0., 0.), std: _triplet_T=(1., 1., 1.)):
        # pylint: disable-msg=not-callable
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, img: torch.Tensor, bboxes: _bboxes_T):
        device = img.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        img.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return img, bboxes

class DeNormalize:

    def __init__(self, mean: _triplet_T=(0., 0., 0.), std: _triplet_T=(1., 1., 1.)):
        # pylint: disable-msg=not-callable
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, img: torch.Tensor, bboxes: _bboxes_T):
        device = img.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        img.mul_(self.std[:, None, None]).add_(self.mean[:, None, None])
        return img, bboxes

class Resize:

    def __init__(self, size: Union[_size_T, Callable[[], _size_T]], pad_val: float=0.5):
        self.pad_val = pad_val
        self.size = size

    def __call__(self, img: torch.Tensor, bboxes: _bboxes_T):
        if callable(self.size):
            target_h, target_w = self.size()
        else:
            target_h, target_w = self.size
        _, img_h, img_w = img.shape

        resize_ratio = min(target_w / img_w, target_h / img_h)
        resize_w = round(resize_ratio * img_w)
        resize_h = round(resize_ratio * img_h)
        image_resized = F.interpolate(img.unsqueeze_(0), size=(resize_h, resize_w),
            mode='bilinear', align_corners=True).squeeze_(0)

        dl = (target_w - resize_w) // 2
        dr = target_w - resize_w - dl
        du = (target_h - resize_h) // 2
        dd = target_h - resize_h - du
        img_padded = F.pad(image_resized, pad=(dl, dr, du, dd), value=self.pad_val)

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

    def __call__(self, img: torch.Tensor, bboxes: _bboxes_T):
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
    
    def __call__(self, img, bboxes: _bboxes_T):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img/255., (2, 0, 1)).astype(np.float32)
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
