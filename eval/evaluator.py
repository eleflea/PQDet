from collections import defaultdict, namedtuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import tools
from config import size_fix

Label = namedtuple('Label', ['bboxes', 'seen', 'difficult'])
MAP = namedtuple('MAP', ['mean', 'classes'])

def convert_pred(batch_pred_bbox: torch.Tensor,
    input_size: torch.Tensor,
    batch_original_size: torch.Tensor) -> torch.Tensor:
    '''
    batch_pred_bbox: (B, ?, C+5)
    input_size: (2, )
    batch_original_size: (B, 2) or (2, )
    '''
    num_classes = batch_pred_bbox.shape[-1] - 5
    # (B, ?, 4) (B, ?, 1) (B, ?, C)
    pred_coor, pred_conf, pred_prob = batch_pred_bbox.split([4, 1, num_classes], dim=-1)

    resize_ratio, _ = (input_size/batch_original_size).min(dim=-1) # (B, )
    delta = (input_size - resize_ratio.unsqueeze_(-1) * batch_original_size) / 2 # (B, 2)
    pred_coor\
        .sub_(delta[..., [1, 0]].repeat(1, 2).unsqueeze_(1))\
        .div_(resize_ratio.unsqueeze_(-1)) # (B, ? , 4)

    bossl = len(batch_original_size.shape)
    if bossl == 2:
        max_edge = (batch_original_size-1)[..., [1, 0]].unsqueeze_(1)
    elif bossl == 1:
        max_edge = (batch_original_size-1)[..., [1, 0]]
    pred_coor[..., :2].clamp_min_(0)
    pred_coor[..., 2:] = torch.min(pred_coor[..., 2:], max_edge)

    pred_prob.mul_(pred_conf)

    bboxes = torch.cat([pred_coor, pred_prob], dim=-1) # (B, ?, 4+C)
    return bboxes

class Evaluator:

    def __init__(self, model: nn.DataParallel, dataset: DataLoader, config):
        self._score_threshold = config.eval.score_threshold
        self._iou_threshold = config.eval.iou_threshold
        self._map_iou = config.eval.map_iou
        self._input_size = size_fix(config.eval.input_size)
        self._classes = config.dataset.classes

        self.model = model
        self.dataset = dataset
        self.init_statics()

    def init_statics(self):
        PQ_func = lambda: tools.PriorityQueue(key=lambda x: -x[1][4])
        self.detections = defaultdict(PQ_func)
        self.labels = defaultdict(dict)
        self.gt_count = defaultdict(int)

    def predict(self, imgs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_pred_bbox = self.model(imgs)
        return batch_pred_bbox

    def evaluate(self) -> MAP:
        for data in tqdm(self.dataset):
            batch_img, batch_file_name, batch_img_shape,\
                batch_label, batch_diff = data
            batch_pred_bbox = self.predict(batch_img)
            device = batch_pred_bbox.device
            input_size = torch.FloatTensor(self._input_size).to(device)
            batch_img_shape = batch_img_shape.to(device)
            batch_pred_bbox = convert_pred(batch_pred_bbox, input_size, batch_img_shape)
            for file_name, labels, diffs, pred_bboxes\
                in zip(batch_file_name, batch_label, batch_diff, batch_pred_bbox):
                bboxes = tools.torch_nms(
                    pred_bboxes,
                    self._score_threshold,
                    self._iou_threshold
                ).cpu().numpy()
                self.add_detections(file_name, bboxes)
                self.add_labels(file_name, labels, diffs)
        return self.mAP()

    def mAP(self) -> MAP:
        mAP_class = {}
        for class_index, detections in self.detections.items():
            class_index = int(class_index)
            tp = np.zeros(len(detections))
            fp = np.zeros(len(detections))
            for i, (file_name, bbox) in enumerate(detections):
                label = self.labels[file_name].get(class_index)
                iou_max = -np.inf
                if label is not None:
                    BBGT = label.bboxes
                    bb = bbox[:4]
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    iou_max = np.max(overlaps)
                    j_max = np.argmax(overlaps)
                    # ious = tools.iou_calc1(bbox[:4], label.bboxes)
                    # iou_max = np.max(ious)
                    # j_max = np.argmax(ious)
                if iou_max > self._map_iou:
                    if label.difficult[j_max]:
                        continue
                    if label.seen[j_max]:
                        fp[i] = 1
                    else:
                        tp[i] = 1
                        label.seen[j_max] = True
                else:
                    fp[i] = 1
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / self.gt_count[class_index]
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            mAP_class[self._classes[class_index]] = self.ap(rec, prec)
        mAP = MAP(np.mean(list(mAP_class.values())), mAP_class)
        self.init_statics()
        return mAP

    @staticmethod
    def ap(rec: np.ndarray, prec: np.ndarray) -> float:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        ap = np.sum(np.diff(mrec) * mpre[1:])
        return ap

    def add_detections(self, file_name: str, bboxes: np.ndarray):
        for bbox in bboxes:
            self.detections[int(bbox[-1])].push((file_name, bbox))

    def add_labels(self, file_name: str, bboxes: np.ndarray, diffs: np.ndarray):
        classes = bboxes[:, -1].astype(int)
        for class_index in set(classes):
            select_indeces = classes == class_index
            select_bboxes = bboxes[select_indeces][:, :4]
            select_diffs = diffs[select_indeces].astype(np.bool)
            seens = [False for _ in range(len(select_bboxes))]
            self.labels[file_name][class_index] = Label(select_bboxes, seens, select_diffs)
            self.gt_count[class_index] += np.sum(~select_diffs)
