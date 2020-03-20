from math import ceil

import numpy as np
import torch
from torch.utils.data import Dataset

from config import size_fix
from dataset import augment
from dataset.sample import SampleGetter


class EvalDataset(Dataset):

    def __init__(self, config):
        self._dataset_file = config.dataset.eval_txt_file
        self._input_size = size_fix(config.eval.input_size)
        self._batch_size = config.eval.batch_size
        self._classes = config.dataset.classes
        self._partial_num = None if config.eval.partial == 0 else config.eval.partial

        self.sample_getter = SampleGetter(self._classes, mode='eval')

        with open(self._dataset_file, 'r') as fr:
            self._imgs = [line.strip() for line in fr.readlines() if len(line.strip()) != 0][:self._partial_num]
        self._num_imgs = len(self._imgs)

        self.augment = augment.Compose([
            augment.Resize(self._input_size),
            augment.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            augment.ToTensor('cpu'),
        ])

    def __len__(self):
        return ceil(self._num_imgs / self._batch_size)

    @property
    def length(self):
        return self._num_imgs

    def get_sample(self, img_index: int):
        image, label = self.sample_getter(self._imgs[img_index])
        image, _ = self.augment(image, [])
        return (image, *label)

    def __getitem__(self, index: int):
        if index >= len(self): raise StopIteration
        batch_image = []
        batch_file_name, batch_shape, batch_bbox, batch_diff = [], [], [], []

        end = min(self._num_imgs, (index + 1) * self._batch_size)
        for idx in range(index*self._batch_size, end):
            image, file_name, shape, bboxes, diffs = self.get_sample(idx)
            batch_image.append(image)
            batch_file_name.append(file_name)
            batch_shape.append(shape)
            batch_bbox.append(bboxes)
            batch_diff.append(diffs)

        return torch.stack(batch_image), batch_file_name,\
            torch.from_numpy(np.stack(batch_shape)),\
            batch_bbox, batch_diff
