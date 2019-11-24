import logging
import os
import random
from os import path
from math import ceil

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config as cfg
import dataset.data_aug as data_aug
import tools


def get_sample_by_img_path(img_path):
    image = np.array(cv2.imread(img_path))
    height, width, _ = image.shape
    # only voc dataset
    label_path = img_path.replace('JPEGImages', 'labels').replace('.jpg', '.txt')
    # yolo label style: class_id xc yc w h
    # turn it to: [xmin, ymin, xmax, ymax, class_id]
    temp_bboxes = []
    with open(label_path, 'r') as fr:
        for line in fr.readlines():
            if len(line.strip()) == 0:
                continue
            nums = [float(num) for num in line.strip().split(' ')]
            temp_bboxes.append([
                round((nums[1]-nums[3]/2)*width),
                round((nums[2]-nums[4]/2)*height),
                round((nums[1]+nums[3]/2)*width),
                round((nums[2]+nums[4]/2)*height),
                int(nums[0])
            ])
    bboxes = np.array(temp_bboxes)
    return image, bboxes

class YOLODataset(Dataset):

    def __init__(self):
        self.__dataset_file = cfg.TRAIN_DATASET_FILE
        self.__train_input_sizes = cfg.TRAIN_INPUT_SIZES
        self.__strides = np.array(cfg.STRIDES)
        self.__batch_size = cfg.BATCH_SIZE
        self.__classes = cfg.CLASSES
        self.__num_classes = len(self.__classes)
        self.__gt_per_grid = cfg.GT_PER_GRID
        self.__class_to_ind = dict(zip(self.__classes, range(self.__num_classes)))

        with open(self.__dataset_file, 'r') as fr:
            self.imgs = [line.strip() for line in fr.readlines() if len(line.strip()) != 0]
        self.__num_imgs = len(self.imgs)

        print(f'{self.__num_imgs} for train.')

    def __len__(self):
        return ceil(self.__num_imgs / self.__batch_size)

    def __get_sample(self, img_path, size):
        image, bboxes = get_sample_by_img_path(img_path)

        image, bboxes = data_aug.random_horizontal_flip(image, bboxes)
        image, bboxes = data_aug.random_crop(image, bboxes)
        image, bboxes = data_aug.random_translate(image, bboxes)
        image, bboxes = tools.img_preprocess2(image, bboxes, (size, size), True)
        return image, bboxes

    def __getitem__(self, index):
        random_trainsize = random.choice(self.__train_input_sizes)
        outputshapes = random_trainsize // self.__strides

        batch_image = np.zeros((self.__batch_size, 3, random_trainsize, random_trainsize))
        batch_label_sbbox = np.zeros((self.__batch_size, outputshapes[0], outputshapes[0],
                                      self.__gt_per_grid, 6 + self.__num_classes))
        batch_label_mbbox = np.zeros((self.__batch_size, outputshapes[1], outputshapes[1],
                                      self.__gt_per_grid, 6 + self.__num_classes))
        batch_label_lbbox = np.zeros((self.__batch_size, outputshapes[2], outputshapes[2],
                                      self.__gt_per_grid, 6 + self.__num_classes))
        temp_batch_sbboxes = []
        temp_batch_mbboxes = []
        temp_batch_lbboxes = []
        max_sbbox_per_img = 0
        max_mbbox_per_img = 0
        max_lbbox_per_img = 0
        for idx in range(self.__batch_size):
            img_index = index * self.__batch_size + idx
            if img_index >= self.__num_imgs:
                img_path = self.imgs[random.randint(0, self.__num_imgs - 1)]
            else:
                img_path = self.imgs[img_index]
            image, bboxes = self.__get_sample(img_path, random_trainsize)

            # mixup
            if random.random() < 0.5:
                index_mix = random.randint(0, self.__num_imgs - 1)
                image_mix, bboxes_mix = self.__get_sample(self.imgs[index_mix], random_trainsize)

                lam = np.random.beta(1.5, 1.5)
                image = lam * image + (1 - lam) * image_mix
                bboxes = np.concatenate(
                    [bboxes, np.full((len(bboxes), 1), lam)], axis=-1)
                bboxes_mix = np.concatenate(
                    [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=-1)
                bboxes = np.concatenate([bboxes, bboxes_mix])
            else:
                bboxes = np.concatenate([bboxes, np.full((len(bboxes), 1), 1.0)], axis=-1)

            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__create_label(bboxes, outputshapes)
            batch_image[idx, :, :, :] = image
            batch_label_sbbox[idx, :, :, :, :] = label_sbbox
            batch_label_mbbox[idx, :, :, :, :] = label_mbbox
            batch_label_lbbox[idx, :, :, :, :] = label_lbbox

            zeros = np.zeros((1, 4), dtype=np.float32)
            sbboxes = sbboxes if len(sbboxes) != 0 else zeros
            mbboxes = mbboxes if len(mbboxes) != 0 else zeros
            lbboxes = lbboxes if len(lbboxes) != 0 else zeros
            temp_batch_sbboxes.append(sbboxes)
            temp_batch_mbboxes.append(mbboxes)
            temp_batch_lbboxes.append(lbboxes)
            max_sbbox_per_img = max(max_sbbox_per_img, len(sbboxes))
            max_mbbox_per_img = max(max_mbbox_per_img, len(mbboxes))
            max_lbbox_per_img = max(max_lbbox_per_img, len(lbboxes))

        batch_sbboxes = np.array(
            [np.concatenate([sbboxes, np.zeros((max_sbbox_per_img + 1 - len(sbboxes), 4), dtype=np.float32)], axis=0)
             for sbboxes in temp_batch_sbboxes])
        batch_mbboxes = np.array(
            [np.concatenate([mbboxes, np.zeros((max_mbbox_per_img + 1 - len(mbboxes), 4), dtype=np.float32)], axis=0)
             for mbboxes in temp_batch_mbboxes])
        batch_lbboxes = np.array(
            [np.concatenate([lbboxes, np.zeros((max_lbbox_per_img + 1 - len(lbboxes), 4), dtype=np.float32)], axis=0)
             for lbboxes in temp_batch_lbboxes])
        return torch.from_numpy(np.array(batch_image).astype(np.float32)),\
               torch.from_numpy(np.array(batch_label_sbbox).astype(np.float32)), \
               torch.from_numpy(np.array(batch_label_mbbox).astype(np.float32)), \
               torch.from_numpy(np.array(batch_label_lbbox).astype(np.float32)), \
               torch.from_numpy(np.array(batch_sbboxes).astype(np.float32)), \
               torch.from_numpy(np.array(batch_mbboxes).astype(np.float32)), \
               torch.from_numpy(np.array(batch_lbboxes).astype(np.float32))

    def __create_label(self, bboxes, train_output_sizes):
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
        label = [np.zeros((train_output_sizes[i], train_output_sizes[i],
            self.__gt_per_grid, 6 + self.__num_classes)) for i in range(3)]
        # mixup weight位默认为1.0
        for i in range(3):
            label[i][:, :, :, -1] = 1.0
        bboxes_coor = [[] for _ in range(3)]
        bboxes_count = [np.zeros((train_output_sizes[i], train_output_sizes[i])) for i in range(3)]

        for bbox in bboxes:
            # (1)获取bbox在原图上的顶点坐标、类别索引、mix up权重、中心坐标、高宽、尺度
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mixw = bbox[5]
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_scale = np.sqrt(np.multiply.reduce(bbox_xywh[2:]))

            # label smooth
            onehot = np.zeros(self.__num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.__num_classes, 1.0 / self.__num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            if bbox_scale <= 30:
                match_branch = 0
            elif 30 < bbox_scale <= 90:
                match_branch = 1
            else:
                match_branch = 2

            xind, yind = np.floor(1.0 * bbox_xywh[:2] / self.__strides[match_branch]).astype(np.int32)
            gt_count = int(bboxes_count[match_branch][yind, xind])
            if gt_count < self.__gt_per_grid:
                if gt_count == 0:
                    gt_count = slice(None)
                bbox_label = np.concatenate([bbox_coor, [1.0], smooth_onehot, [bbox_mixw]], axis=-1)
                label[match_branch][yind, xind, gt_count, :] = 0
                label[match_branch][yind, xind, gt_count, :] = bbox_label
                bboxes_count[match_branch][yind, xind] += 1
                bboxes_coor[match_branch].append(bbox_coor)
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_coor
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
