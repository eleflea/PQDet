import random
from math import ceil, pi

import numpy as np
import torch
from torch.utils.data import Dataset

from config import sizes_fix
from dataset import augment
from dataset.sample import SampleGetter


class TrainDataset(Dataset):

    def __init__(self, config):
        self._dataset_file = config.dataset.train_txt_file
        self._input_sizes = sizes_fix(config.train.input_sizes)
        self._strides = np.array(config.model.strides)
        self._batch_size = config.train.batch_size
        self._classes = config.dataset.classes
        self._num_classes = len(self._classes)
        self._gt_per_grid = config.model.gt_per_grid
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(self._dataset_file, 'r') as fr:
            self._imgs = [line.strip() for line in fr.readlines() if len(line.strip()) != 0]
        self._num_imgs = len(self._imgs)

        self.sample_getter = SampleGetter(self._classes, train=True)

        max_index = np.argmax([h*w for h, w in self._input_sizes])
        self.input_size = self._input_sizes[max_index]
        self.random_size = self._bubble

        self.mixup = augment.Mixup(self.sample_sample, beta=1.5)
        self.augment = augment.Compose([
            augment.ToTensor(self._device),
            augment.RandomHFlip(p=0.5),
            augment.RandomCrop(p=0.5),
            augment.ColorJitter(
                brightness=[-0.1, 0.1],
                contrast=[0.8, 1.2],
                saturation=[0.2, 1.5],
                hue=[-pi/3, pi/3],
            ),
            augment.Resize(self._get_input_size),
            augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return ceil(self._num_imgs / self._batch_size)
    
    @property
    def length(self):
        return self._num_imgs

    def _get_input_size(self):
        return self.input_size

    def _bubble(self):
        self.random_size = self.random_size_

    def random_size_(self):
        self.input_size = random.choice(self._input_sizes)

    def get_sample(self, img_index: int):
        image, bboxes = self.sample_getter(self._imgs[img_index])
        image, bboxes = self.augment(image, bboxes)
        return image, bboxes

    def sample_sample(self):
        idx = random.randint(0, self._num_imgs - 1)
        return self.get_sample(idx)

    @staticmethod
    def _npzeros(*dims):
        return np.zeros(dims, dtype=np.float32)

    def __getitem__(self, index):
        self.random_size()
        output_sizes = self.input_size // self._strides[:, None]

        batch_image = []
        batch_label_sbbox, batch_label_mbbox, batch_label_lbbox = [], [], []
        temp_batch_sbboxes, temp_batch_mbboxes, temp_batch_lbboxes = [], [], []
        max_sbbox_per_img, max_mbbox_per_img, max_lbbox_per_img = 0, 0, 0

        for idx in range(index*self._batch_size, (index+1)*self._batch_size):
            if idx < self._num_imgs:
                image, bboxes = self.get_sample(idx)
            else:
                image, bboxes = self.sample_sample()

            image, bboxes = self.mixup(image, bboxes)

            label_sbbox, label_mbbox, label_lbbox,\
                sbboxes, mbboxes, lbboxes = self.create_label(bboxes, output_sizes)
            batch_image.append(image)
            batch_label_sbbox.append(label_sbbox)
            batch_label_mbbox.append(label_mbbox)
            batch_label_lbbox.append(label_lbbox)

            zeros = self._npzeros(1, 4)
            sbboxes = sbboxes if len(sbboxes) != 0 else zeros
            mbboxes = mbboxes if len(mbboxes) != 0 else zeros
            lbboxes = lbboxes if len(lbboxes) != 0 else zeros
            temp_batch_sbboxes.append(sbboxes)
            temp_batch_mbboxes.append(mbboxes)
            temp_batch_lbboxes.append(lbboxes)
            max_sbbox_per_img = max(max_sbbox_per_img, len(sbboxes))
            max_mbbox_per_img = max(max_mbbox_per_img, len(mbboxes))
            max_lbbox_per_img = max(max_lbbox_per_img, len(lbboxes))

        batch_sbboxes = [np.concatenate(
            [sbboxes, self._npzeros(max_sbbox_per_img + 1 - len(sbboxes), 4)], axis=0)
                for sbboxes in temp_batch_sbboxes]
        batch_mbboxes = [np.concatenate(
            [mbboxes, self._npzeros(max_mbbox_per_img + 1 - len(mbboxes), 4)], axis=0)
                for mbboxes in temp_batch_mbboxes]
        batch_lbboxes = [np.concatenate(
            [lbboxes, self._npzeros(max_lbbox_per_img + 1 - len(lbboxes), 4)], axis=0)
                for lbboxes in temp_batch_lbboxes]
        return torch.stack(batch_image),\
            torch.from_numpy(np.stack(batch_label_sbbox)).to(self._device),\
            torch.from_numpy(np.stack(batch_label_mbbox)).to(self._device),\
            torch.from_numpy(np.stack(batch_label_lbbox)).to(self._device),\
            torch.from_numpy(np.stack(batch_sbboxes)).to(self._device),\
            torch.from_numpy(np.stack(batch_mbboxes)).to(self._device),\
            torch.from_numpy(np.stack(batch_lbboxes)).to(self._device)

    def create_label(self, bboxes, output_sizes):
        """
        :param bboxes: 一张图对应的所有bbox和每个bbox所属的类别，以及mixup的权重，
        bbox的坐标为(xmin, ymin, xmax, ymax, class_ind, mixup_weight)
        :return:
        label_sbbox: shape为(anchor_per_scale, 6 + num_classes, input_size / 8, input_size / 8)
        label_mbbox: shape为(anchor_per_scale, 6 + num_classes, input_size / 16, input_size / 16)
        label_lbbox: shape为(anchor_per_scale, 6 + num_classes, input_size / 32, input_size / 32)
        只要某个GT落入grid中，那么这个grid就负责预测它，最多负责预测gt_per_grid个GT，
        那么该grid中对应位置的数据为(xmin, ymin, xmax, ymax, 1, classes, mixup_weights),
        其他grid对应位置的数据都为(0, 0, 0, 0, 0, 0..., 1)
        sbboxes：shape为(max_bbox_per_scale, 4)
        mbboxes：shape为(max_bbox_per_scale, 4)
        lbboxes：shape为(max_bbox_per_scale, 4)
        存储的坐标为(xmin, ymin, xmax, ymax)，大小都是bbox纠正后的原始大小
        """
        def gauss_dist(m, x, y, w, h):
            '''m: (y, x)'''
            return np.exp(-0.5*(np.square((m[..., 1]-x)/w*3) + np.square((m[..., 0]-y)/h*3)))

        def mesh_grid(x, y):
            xaxis, yaxis = np.meshgrid(np.arange(x), np.arange(y))
            return np.stack([yaxis, xaxis], axis=-1)

        # grid = [mesh_grid(output_sizes[i], output_sizes[i]) for i in range(3)]

        label = [self._npzeros(output_sizes[i][0], output_sizes[i][1],
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
