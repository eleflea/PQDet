from typing import Sequence, Tuple, Union

import torch
from yacs.config import CfgNode as CN


def size_fix(size):
    '''修正尺寸。如果输入是单个int，则变为相同的二元组。
    如果输入是二元组，则不变。
    '''
    if isinstance(size, int):
        return (size, size)
    return size

def sizes_fix(sizes):
    '''修正一系列尺寸，对列表中的尺寸进行修正。详见`size_fix`。
    '''
    return [size_fix(size) for size in sizes]

def get_device(gpus):
    if gpus is None or len(gpus) == 0:
        return torch.device('cpu')
    return torch.device('cuda')

def fix_gpus(gpus):
    if len(gpus) == 0:
        return None
    return gpus

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
_C.dataset.train_txt_file: str = '/home/eleflea/dataset/Pascal_voc/train.txt'
# 验证数据txt文件，yolo like
_C.dataset.eval_txt_file: str = '/home/eleflea/dataset/Pascal_voc/2007_test.txt'
# 数据集类别名称
_C.dataset.classes: Sequence[str] = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

_C.model = CN()
# 模型定义文件路径
_C.model.cfg_path: str = 'model/cfg/mobilenetv2-yolo.cfg'
_C.model.strides: Sequence[int] = [8, 16, 32]
_C.model.gt_per_grid: int = 3

_C.train = CN()
# 训练时输入图像大小，List[int]或List[Tuple[int, int]]
# 前者长宽相等，后者两个数代表(height, width)
sizes_T = Union[Sequence[int], Sequence[Tuple[int, int]]]
_C.train.input_sizes: sizes_T = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
# 训练batch size
_C.train.batch_size: int = 12
# 训练学习率规划器，'cosine'或'step'
_C.train.scheduler: str = 'cosine'
# 训练初始学习率
_C.train.learning_rate_init: float = 2e-4
# 训练结束时学习率
_C.train.learning_rate_end: float = 1e-6
# step学习率规划器的里程碑
_C.train.mile_stones: Sequence[int] = [30, 45]
# step学习率规划器的gamma
_C.train.gamma: float = 0.1
# 训练热身轮数
_C.train.warmup_epochs: int = 1
# 训练最大轮数
_C.train.max_epochs: int = 80

_C.augment = CN()
# 数据增强中mix up方法应用的概率
_C.augment.mixup_p: float = 0.5
# 数据增强中color jitter方法应用的概率
_C.augment.color_p: float = 1
# 数据增强中horizon flip方法应用的概率
_C.augment.hflip_p: float = 0.5
# 数据增强中random crop方法应用的概率
_C.augment.crop_p: float = 0.75

_C.weight = CN()
# 权重存储路径
_C.weight.dir: str = 'weights'
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
_C.eval.input_size: Union[int, Tuple[int, int]] = 512
# 评估时batch size
_C.eval.batch_size: int = 16
# 评估时置信度阈值
_C.eval.score_threshold: float = 0.1
# NMS时IOU阈值
_C.eval.iou_threshold: float = 0.45
# mAP的IOU阈值
_C.eval.map_iou: float = 0.5
# 评估时使用前一部分图像以加快速度，为0时使用全部
_C.eval.partial: int = 0

_C.sparse = CN()
# 是否进行稀疏化训练
_C.sparse.switch: bool = False
# 稀疏化训练系数
_C.sparse.ratio: float = 0.01

_C.prune = CN()
# 剪枝权重路径，为''时不进行剪枝
_C.prune.weight: str = 'weights/VOC_prune/model-38-0.7486.pt'
# 剪枝后新cfg文件存储路径
_C.prune.new_cfg: str = 'model/cfg/myolo-prune.cfg'
# 剪枝率
_C.prune.ratio: float = 0.3

_C.quant = CN()
# 是否进行QAT
_C.quant.switch: bool = False
# QAT后端 'fbgemm' or 'qnnpack'
_C.quant.backend: str = 'fbgemm'
# QAT时在多少轮之后关闭observer
_C.quant.disable_observer_after: int = 4
# QAT时在多少轮以后冻结BN层参数
_C.quant.freeze_bn_after: int = 8

cfg = _C
