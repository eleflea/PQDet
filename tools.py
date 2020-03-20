import torch
from time import time_ns
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar, Callable, Optional
import heapq


def iou_calc1(boxes1: np.ndarray, boxes2: np.ndarray):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = inter_area / np.maximum(union_area, 1e-14)
    return IOU

def iou_calc3(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = inter_area / union_area
    return IOU

def giou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    intersection_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    intersection_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    intersection = torch.max(intersection_right_down - intersection_left_up, torch.zeros_like(intersection_right_down))
    inter_area = intersection[..., 0] * intersection[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = inter_area / union_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_left_up))
    enclose_area = enclose[..., 0] * enclose[..., 1]
    GIOU = IOU - (enclose_area - union_area) / enclose_area

    return GIOU

def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes:
    假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    :return: best_bboxes
    假设NMS后剩下N个bbox，那么best_bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = iou_calc1(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    return np.array(best_bboxes)

def torch_nms(bboxes: torch.Tensor, score_threshold: float,
    iou_threshold: float, sigma: float=0.3, method: str='nms') -> torch.Tensor:
    """
    :param bboxes:
    假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    :return: best_bboxes
    假设NMS后剩下N个bbox，那么best_bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    """
    device = bboxes.device
    class_scores = bboxes[:, 4:]
    num_classes = class_scores.shape[1]
    best_bboxes = []

    for class_index in range(num_classes):
        scores = class_scores[:, class_index]
        pick_indeces = (scores > score_threshold).nonzero().squeeze(-1)
        if len(pick_indeces) == 0:
            continue
        pick_bboxes = torch.cat([
            bboxes[:, :4][pick_indeces],
            scores[pick_indeces][:, None],
            torch.full((len(pick_indeces), 1), class_index, dtype=torch.float32).to(device)
        ], dim=1)
        while len(pick_bboxes) > 0:
            max_ind = torch.argmax(pick_bboxes[:, 4])
            best_bbox = pick_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            pick_bboxes = torch.cat([pick_bboxes[:max_ind], pick_bboxes[max_ind + 1:]])
            iou = iou_calc3(best_bbox[None, :4], pick_bboxes[:, :4])
            assert method in {'nms', 'soft-nms'}
            weight = torch.ones_like(iou)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = torch.exp(-(1.0 * iou ** 2 / sigma))
            pick_bboxes[:, 4].mul_(weight)
            score_mask = pick_bboxes[:, 4] > score_threshold
            pick_bboxes = pick_bboxes[score_mask]
    # pylint: disable-msg=not-callable
    return torch.stack(best_bboxes) if len(best_bboxes) > 0 else torch.tensor([]).to(device)

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, temp_sum, n=1):
        self.sum += temp_sum
        self.count += n

    def get_avg_reset(self):
        if self.count == 0:
            return 0.
        avg = float(self.sum) / float(self.count)
        self.reset()
        return avg

    def get_sum_reset(self):
        s = self.sum
        self.sum = 0
        return s

class TicToc:

    def __init__(self, name: Optional[str]=None):
        self.name = name
        self.last = 0
        self.records = []
        self.reset()

    def reset(self):
        self.last = 0
        self.records.clear()

    def tic(self):
        self.last = time_ns()

    def toc(self):
        self.records.append(time_ns() - self.last)

    def __getitem__(self, index):
        return self.records[index]

    def mean(self):
        return np.mean(self.records)

    def mean_reset(self):
        m = self.mean()
        self.reset()
        return m

    def sum(self):
        return np.sum(self.records)

    def sum_reset(self):
        s = self.sum()
        self.reset()
        return s
    
    def statistics(self):
        std = np.std(self.records)
        return {
            'name': 'none' if self.name is None else self.name,
            'mean': np.mean(self.records),
            'std': std,
            '3std': 3*std,
            'min': np.amin(self.records),
            'max': np.amax(self.records),
        }

class _Comparable(metaclass=ABCMeta):
    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...

    @abstractmethod
    def __neg__(self): ...

_comparable_T = TypeVar('Comparable_T', bound=_Comparable)
_key_func_T = Optional[Callable[[Any], _comparable_T]]
_val_T = TypeVar('val_T')

class PriorityQueue:

    def __init__(self, key: _key_func_T):
        self.key = (lambda x: x) if key is None else key
        self.queue = []
        self._index = 0

    def push(self, val: _val_T):
        priority = self.key(val)
        heapq.heappush(self.queue, (priority, self._index, val))
        self._index += 1

    def pop(self) -> _val_T:
        return heapq.heappop(self.queue)[-1]

    def __len__(self):
        return len(self.queue)

    def __next__(self):
        try:
            return self.pop()
        except IndexError:
            raise StopIteration

    def __iter__(self):
        return self