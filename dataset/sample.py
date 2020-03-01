from typing import Sequence
from xml.etree.ElementTree import parse

import numpy as np
import cv2

class SampleGetter:

    def __init__(self, classes: Sequence[str], train: bool=True):
        self.classes = classes
        self.parser = VOCXmlParser(self.classes, train)

    def __call__(self, img_path: str):
        image = cv2.imread(img_path)
        label_path = img_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
        label = self.parser(label_path)
        return image, label

class VOCXmlParser:

    def __init__(self, classes: Sequence[str], train: bool):
        self.classes = classes
        self.cls_to_idx = dict(zip(classes, range(len(classes))))
        self.train = train

    def __call__(self, file_name):
        root = parse(file_name).getroot()
        if self.train:
            return self.boxes(root)
        else:
            return (self.file_name(root), self.shape(root), *self.boxes(root))

    def file_name(self, root):
        return root.find('filename').text

    def shape(self, root):
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        return np.array([h, w], dtype=np.float32)

    def boxes(self, root):
        bbs, diffs = [], []
        obj_tags = root.findall('object')
        for t in obj_tags:
            diff = int(t.find('difficult').text)
            if self.train and diff == 1:
                continue
            cls_name = t.find('name').text
            cls_idx = self.cls_to_idx[cls_name]
            box_tag = t.find('bndbox')
            x1 = box_tag.find('xmin').text
            y1 = box_tag.find('ymin').text
            x2 = box_tag.find('xmax').text
            y2 = box_tag.find('ymax').text
            box = [float(x1), float(y1), float(x2), float(y2), cls_idx]
            bbs.append(box)
            diffs.append(diff)
        bbs = np.array(bbs, dtype=np.float32)
        if self.train:
            return bbs
        return bbs, np.array(diffs)