from math import ceil

import numpy as np
import torch
from torch.utils.data import Dataset

from config import size_fix
from dataset import SAMPLE_GETTER_REGISTER


class EvalDataset(Dataset):

    def __init__(self, config):
        self._dataset_name = config.dataset.name.lower()
        self._dataset_file = config.dataset.eval_txt_file
        self._input_size = size_fix(config.eval.input_size)
        self._batch_size = config.eval.batch_size
        self._classes = config.dataset.classes
        self._partial_num = None if config.eval.partial == 0 else config.eval.partial

        self.sample_getter = SAMPLE_GETTER_REGISTER[self._dataset_name](
            mode='eval', classes=self._classes
        ).set_eval_augment(self._input_size)

        with open(self._dataset_file, 'r') as fr:
            self._imgs = [line.strip() for line in fr.readlines() if len(line.strip()) != 0][:self._partial_num]
        self._num_imgs = len(self._imgs)

    def __len__(self):
        return ceil(self._num_imgs / self._batch_size)

    @property
    def length(self):
        return self._num_imgs

    def __getitem__(self, index: int):
        if index >= len(self): raise StopIteration
        batch_image = []
        batch_file_name, batch_shape, batch_bbox, batch_diff = [], [], [], []

        end = min(self._num_imgs, (index + 1) * self._batch_size)
        for idx in range(index*self._batch_size, end):
            image, file_name, shape, bboxes, diffs = self.sample_getter(self._imgs[idx])
            batch_image.append(image)
            batch_file_name.append(file_name)
            batch_shape.append(shape)
            batch_bbox.append(bboxes)
            batch_diff.append(diffs)

        return torch.stack(batch_image), batch_file_name,\
            torch.from_numpy(np.stack(batch_shape)),\
            batch_bbox, batch_diff
