import random
from math import ceil

import numpy as np
import torch
from torch.utils.data import Dataset

import tools
from config import sizes_fix
from dataset import augment
from dataset.sample import SampleGetter


def _npzeros(*dims):
    return np.zeros(dims, dtype=np.float32)

def _pad_to_size(array, size: int):
    la = len(array)
    if la == 0:
        return _npzeros(size, 4)
    return np.pad(array, ((0, size-la), (0, 0)), 'constant', constant_values=0)

def _pad_arrays(arrays):
    max_length = max(max(len(array) for array in arrays), 1)
    return [_pad_to_size(array, max_length) for array in arrays]

def collate_batch(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, np.ndarray):
        return collate_batch([torch.as_tensor(b) for b in batch])
    else:
        transposed = list(zip(*batch))
        images_and_labels = [collate_batch(samples) for samples in transposed[:4]]
        bboxs = [collate_batch(_pad_arrays(samples)) for samples in transposed[4:]]
        return (*images_and_labels, *bboxs)

class TrainDataset(Dataset):

    def __init__(self, config):
        self._dataset_file = config.dataset.train_txt_file
        self._input_sizes = sizes_fix(config.train.input_sizes)
        self._strides = np.array(config.model.strides)
        self._batch_size = config.train.batch_size
        self._classes = config.dataset.classes
        self._num_classes = len(self._classes)
        self._gt_per_grid = config.model.gt_per_grid
        self._anchors = np.array(config.model.anchors, dtype=np.float32)

        self._color_p = config.augment.color_p
        self._mixup_p = config.augment.mixup_p
        self._hflip_p = config.augment.hflip_p
        self._crop_p = config.augment.crop_p

        with open(self._dataset_file, 'r') as fr:
            self._imgs = [line.strip() for line in fr.readlines() if len(line.strip()) != 0]
        self._num_imgs = len(self._imgs)

        self.sample_getter = SampleGetter(self._classes, mode='train')

        self.init_shuffle()

        self.mixup = augment.Compose([
            augment.Mixup(self.sample_sample, p=self._mixup_p, beta=1.5),
            augment.ToTensor('cpu'),
        ])
        self.mosaic = augment.Compose([
            augment.Mosaic(self.sample_sample, p=1, size=self._get_input_size),
            augment.Mixup(self.sample_sample, p=0, beta=1.5),
            augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            augment.ToTensor('cpu'),
        ])
        self.augment = augment.Compose([
            augment.RandomHFlip(p=self._hflip_p),
            augment.RandomCrop(p=self._crop_p),
            augment.ColorJitter(
               brightness=[-0.1, 0.1],
               contrast=[0.8, 1.2],
               saturation=[0.1, 2],
               p=self._color_p,
            ),
            augment.Resize(self._get_input_size),
            augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self._length

    @property
    def length(self):
        return self._num_imgs

    def init_shuffle(self):
        batch_len = ceil(self._num_imgs / self._batch_size)
        self._length = batch_len * self._batch_size
        self._shuffle_indexes = random.choices(range(self._num_imgs), k=self._length)
        self._shuffle_sizes = random.choices(self._input_sizes, k=batch_len)
        max_index = np.argmax([h*w for h, w in self._input_sizes])
        self._shuffle_sizes[0] = self.input_size = self._input_sizes[max_index]

    def _get_input_size(self):
        return self.input_size

    def get_sample(self, img_index: int):
        image, bboxes = self.sample_getter(self._imgs[img_index])
        image, bboxes = self.augment(image, bboxes)
        return image, bboxes

    def sample_sample(self):
        idx = random.randint(0, self._num_imgs - 1)
        return self.get_sample(idx)

    def __getitem__(self, index):
        self.input_size = self._shuffle_sizes[index // self._batch_size]
        output_sizes = self.input_size // self._strides[:, None]

        image, bboxes = self.get_sample(self._shuffle_indexes[index])
        # image, bboxes = self.mosaic(image, bboxes)
        image, bboxes = self.mixup(image, bboxes)
        labels = self.create_label3(bboxes, output_sizes)
        return (image, *labels)

    def create_label(self, bboxes, output_sizes):

        def gauss_dist(m, x, y, w, h):
            '''m: (y, x)'''
            return np.exp(-0.5*(np.square((m[..., 1]-x)/w*3) + np.square((m[..., 0]-y)/h*3)))

        def mesh_grid(x, y):
            xaxis, yaxis = np.meshgrid(np.arange(x), np.arange(y))
            return np.stack([yaxis, xaxis], axis=-1)

        # grid = [mesh_grid(output_sizes[i], output_sizes[i]) for i in range(3)]

        label = [_npzeros(output_sizes[i][0], output_sizes[i][1],
            self._gt_per_grid, 6 + self._num_classes) for i in range(3)]
        # mixup weight位默认为1.0
        for i in range(3):
            label[i][..., -1] = 1.0
        bboxes_coor = [[] for _ in range(3)]
        bboxes_count = [np.zeros((output_sizes[i][0], output_sizes[i][1])) for i in range(3)]

        for bbox in bboxes:
            # (1)获取bbox在原图上的顶点坐标、类别索引、mix up权重、中心坐标、高宽、尺度
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mixw = bbox[5]
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_scale = np.sqrt(np.multiply.reduce(bbox_xywh[2:]))

            # label smooth
            onehot = np.zeros(self._num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self._num_classes, 1.0 / self._num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            if bbox_scale <= 30:
                match_branch = 0
            elif 30 < bbox_scale <= 90:
                match_branch = 1
            else:
                match_branch = 2

            xind, yind = np.floor(1.0 * bbox_xywh[:2] / self._strides[match_branch]).astype(np.int32)
            gt_count = int(bboxes_count[match_branch][yind, xind])
            if gt_count < self._gt_per_grid:
                if gt_count == 0:
                    gt_count = slice(None)
                bbox_label = np.concatenate([bbox_coor, [1.0], smooth_onehot, [bbox_mixw]], axis=-1)
                label[match_branch][yind, xind, gt_count, :] = bbox_label
                bboxes_count[match_branch][yind, xind] += 1
                bboxes_coor[match_branch].append(bbox_coor)
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_coor
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def create_label2(self, bboxes, output_sizes):
        label = [_npzeros(output_sizes[i][0], output_sizes[i][1],
            self._gt_per_grid, 6 + self._num_classes) for i in range(3)]
        # mixup weight位默认为1.0
        for i in range(3):
            label[i][..., -1] = 1.0
        bboxes_coor = [[] for _ in range(3)]
        bboxes_count = [_npzeros(output_sizes[i][0], output_sizes[i][1], 3) for i in range(3)]

        for bbox in bboxes:
            # (1)获取bbox在原图上的顶点坐标、类别索引、mix up权重、中心坐标、高宽、尺度
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mixw = bbox[5]
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_scale = np.sqrt(np.multiply.reduce(bbox_xywh[2:]))
            bbox_ratio = bbox_xywh[2] / bbox_xywh[3]

            # label smooth
            onehot = np.zeros(self._num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self._num_classes, 1.0 / self._num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            if bbox_scale <= 30:
                scale_branch = 0
            elif 30 < bbox_scale <= 90:
                scale_branch = 1
            else:
                scale_branch = 2

            if bbox_ratio <= 2/3:
                ratio_branch = 0
            elif 2/3 < bbox_ratio <= 3/2:
                ratio_branch = 1
            else:
                ratio_branch = 2

            xind, yind = np.floor(1.0 * bbox_xywh[:2] / self._strides[scale_branch]).astype(np.int32)
            gt_occupied = bboxes_count[scale_branch][yind, xind, ratio_branch]
            if not gt_occupied:
                bbox_label = np.concatenate([bbox_coor, [1.0], smooth_onehot, [bbox_mixw]], axis=-1)
                label[scale_branch][yind, xind, ratio_branch, :] = bbox_label
                bboxes_count[scale_branch][yind, xind, ratio_branch] = 1
                bboxes_coor[scale_branch].append(bbox_coor)
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_coor
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def create_label3(self, bboxes, output_sizes):
        label = [_npzeros(output_sizes[i][0], output_sizes[i][1],
            self._gt_per_grid, 6 + self._num_classes) for i in range(3)]
        # mixup weight位默认为1.0
        for i in range(3):
            label[i][..., -1] = 1.0
        bboxes_coor = [[] for _ in range(3)]

        for bbox in bboxes:
            # (1)获取bbox在原图上的顶点坐标、类别索引、mix up权重、中心坐标、高宽、尺度
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mixw = bbox[5]
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)

            # label smooth
            onehot = np.zeros(self._num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self._num_classes, 1.0 / self._num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            xy_indexes = (bbox_xywh[:2][:, None]//self._strides).astype(np.int32).T
            xcyc = (xy_indexes.astype(np.float32)+0.5) * self._strides[:, None]
            anchor_bboxes = np.concatenate([np.repeat(xcyc, 3, axis=0), self._anchors], axis=-1)
            ious = tools.iou_xywh_numpy(bbox_xywh, anchor_bboxes)
            iou_mask = ious > 0.3
            if not iou_mask.any():
                # print('dataset: all anchors missed!, highest {:.2f}'.format(ious.max()))
                iou_mask[ious.argmax()] = 1

            for i in iou_mask.nonzero()[0]:
                scale_branch, ratio_branch = i // 3, i % 3
                x, y = xy_indexes[scale_branch]
                bbox_label = np.concatenate([bbox_coor, [1.0], smooth_onehot, [bbox_mixw]], axis=-1)
                label[scale_branch][y, x, ratio_branch, :] = bbox_label
                bboxes_coor[scale_branch].append(bbox_coor)

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_coor
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
