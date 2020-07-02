import glob
from functools import reduce
from itertools import chain
from operator import mul
from os import path

import imagesize
import numpy as np


def repeat_weight_gener(f):
    frac = np.mod(f, 1)
    if frac == 0:
        return lambda: int(f)
    return lambda: int(np.floor(f) + np.random.binomial(1, frac))

sets=['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test']
images_sets = [
    chain(*[glob.glob('%s/images/*.jpg' % s) for s in sets[:2]]), # train&val
    glob.glob('%s/images/*.jpg' % sets[2]), # test
]

# collect area info among trian&val set
area_set = set()
image_path_areas = {}
for file in images_sets[0]:
    area = reduce(mul, imagesize.get(file), 1)
    area_set.add(area)
    image_path_areas[file] = area
sorted_area = sorted(list(area_set))
ratio = [num / sorted_area[0] for num in sorted_area]
area_ratios = dict(zip(sorted_area, [repeat_weight_gener(r) for r in ratio]))

# write list file acorrding calculated repeat weights
with open('trainval.txt', 'w') as fw:
    for img_path, area in image_path_areas.items():
        img_path = path.abspath(img_path)
        repeat_weight = area_ratios[area]()
        fw.write(('%s\n' % img_path) * repeat_weight)
with open('test.txt', 'w') as fw:
    for img_path in images_sets[1]:
        img_path = path.abspath(img_path)
        fw.write('%s\n' % img_path)
