from os import path
from typing import Sequence, Union, Tuple

from yacs.config import CfgNode as CN

def size_fix(size):
    if isinstance(size, int):
        return (size, size)
    return size

def sizes_fix(sizes):
    return [size_fix(size) for size in sizes]

_C = CN()

_C.system = CN()
# GPU设备id，为空时使用CPU
_C.system.gpus: Sequence[int] = (0,)
# 读数据集的线程数
_C.system.num_workers: int = 4

# 实验名称，影响权重存储路径
_C.experiment_name: str = 'VOC'

_C.dataset = CN()
# 训练数据txt文件，yolo like
_C.dataset.train_txt_file: str = '/root/dataset/VOC/train.txt'
# 验证数据txt文件，yolo like
_C.dataset.eval_txt_file: str = '/root/dataset/VOC/2007_test.txt'
# 数据集类别名称
_C.dataset.classes: Sequence[str] = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

_C.model = CN()
_C.model.cfg_path: str = 'model/cfg/mobilenetv2-yolo.cfg'
_C.model.strides: Sequence[int] = [8, 16, 32]
_C.model.gt_per_grid: int = 3

_C.train = CN()
# 训练时输入图像大小，List[int]或List[Tuple[int, int]]
# 前者长宽相等，后者两个数代表(height, width)
sizes_T = Union[Sequence[int], Sequence[Tuple[int, int]]]
_C.train.input_sizes: sizes_T = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
normalizer = max(len(_C.system.gpus), 1)
# 训练batch size
_C.train.batch_size: int = 12 * normalizer
# 训练初始学习率
_C.train.learning_rate_init: float = 1e-4 * _C.train.batch_size / normalizer / 6
# 训练结束时学习率
_C.train.learning_rate_end: float = 1e-6
# 训练热身轮数
_C.train.warmup_epochs: int = 1
# 训练最大轮数
_C.train.max_epochs: int = 80

_C.weight = CN()
# 权重存储路径，受实验名称影响
_C.weight.dir: str = path.join('weights', _C.experiment_name)
# 预训练权重路径，为''时不做迁移学习
_C.weight.backbone: str = 'weights/pretrained/mobilenetv2.pt'
# 恢复训练权重路径，为''时不恢复
_C.weight.resume: str = ''
# 当恢复权重时，是否清除训练历史信息(如已训练的步数)
_C.weight.clear_history: bool = False

_C.eval = CN()
# 自多少轮以后评估指标
_C.eval.after: int = 30
# 评估时输入大小，int或Tuple[int, int]，后者格式为(height, width)
_C.eval.input_size: Union[int, Tuple[int, int]] = 416
# 评估时batch size
_C.eval.batch_size: int = 16
# 评估时置信度阈值
_C.eval.score_threshold: float = 0.01
# NMS时IOU阈值
_C.eval.iou_threshold: float = 0.45
# mAP的IOU阈值
_C.eval.map_iou: float = 0.5

_C.sparse = CN()
# 是否进行稀疏化训练
_C.sparse.on: bool = False
# 稀疏化训练系数
_C.sparse.ratio: float = 0.01

_C.prune = CN()
# 剪枝权重路径，为''时不进行剪枝
_C.prune.weight: str = 'weights/VOC_prune/model-38-0.7486.pt'
# 剪枝率
_C.prune.ratio: float = 0.3

cfg = _C
