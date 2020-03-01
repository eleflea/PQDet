import argparse
import math
import os
from itertools import chain

import torch
import torch.optim as optim
from torch import nn

from config import cfg
from dataset.eval_dataset import EvalDataset
from dataset.train_dataset import TrainDataset
from eval.evaluator import Evaluator
from model.newyolo import YOLOv3
from tools import AverageMeter


def get_bn_modules(model: nn.Module):
    '''遍历整个模型，得到所有需要稀疏化的BN module，暂时是全部。

    Args:
        model: 网络模型。

    Returns:
        list: 所有需要稀疏化的BN module的列表。
    '''
    all_bns = []
    c = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            all_bns.append(m)
            c += 1
    print("sparse mode: {}/{} bns will be sparsed.".format(len(all_bns), c))
    return all_bns

def ensure_weights_dir(path: str):
    # 建立存放模型权重的文件夹（如果不存在的话）
    os.makedirs(path, exist_ok=True)

class Trainer:

    def __init__(self, config):
        # metric
        self.mAP = None
        self.APs = None
        # model
        self._cfg_path = config.model.cfg_path
        # train
        self._train_batch_size = config.train.batch_size
        self._init_lr = config.train.learning_rate_init
        self._end_lr = config.train.learning_rate_end
        self._warmup_epochs = config.train.warmup_epochs
        self._max_epochs = config.train.max_epochs
        # weights
        self._backbone_weight = config.weight.backbone
        self._weights_dir = config.weight.dir
        self._resume_weights = config.weight.resume
        self._clear_history = config.weight.clear_history
        # eval
        self._eval_after = config.eval.after
        # sparse
        self._sparse_train = config.sparse.on
        self._sparse_ratio = config.sparse.ratio
        # system
        self._gpus = config.system.gpus
        self._num_workers = config.system.num_workers
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.init_eopch = 0
        self.global_step = 0
        self.config = config

    def adjust_lr(self, steps: int):
        '''根据训练步数通过公式计算cosine退火的学习率,
        通过对优化器中每个参数赋值调整学习率。

        Args:
            steps: 当前训练的步数。

        Returns:
            float: 学习率。
        '''
        # 热身步数
        warmup_steps = self._warmup_epochs * self._steps_per_epoch
        # 最大训练步数
        max_steps = self._max_epochs * self._steps_per_epoch
        if steps < warmup_steps:
            lr = steps / warmup_steps * self._init_lr
        else:
            lr = self._end_lr + 0.5*(self._init_lr-self._end_lr) *\
                (1 + math.cos((steps-warmup_steps)/(max_steps-warmup_steps)*math.pi))
        # 对每个参数赋值新的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    # 建立数据集
    def init_dataset(self):
        train_dataset = TrainDataset(self.config)
        self.eval_dataset = EvalDataset(self.config)
        # 数据集内部手动生成batch,所以此处batch_size=None
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            num_workers=self._num_workers, pin_memory=True,
        )
        print(f'{train_dataset.length} images for train.')
        print(f'{self.eval_dataset.length} images for evaluate.')

    # 建立YOLOv3模型
    def init_model(self):
        model = YOLOv3(self._cfg_path).to(self._device)
        if len(self._gpus) > 1:
            model = nn.DataParallel(model, device_ids=self._gpus)
        self.model = model

    def init_weights(self):
        # 如果backbone权重存在，载入做迁移学习
        if len(self._backbone_weight) > 0:
            print('loading backbone weights from {}'.format(self._backbone_weight))
            state_dict = torch.load(self._backbone_weight)
            ori_state_dict = self.model.state_dict()
            ori_state_dict.update(state_dict)
            self.model.load_state_dict(ori_state_dict)

        # 如果是恢复训练进度
        if len(self._resume_weights) > 0:
            # 读入待恢复的权重和训练步数
            state_dict = torch.load(self._resume_weights)
            # 如果指定清除训练历史,则训练步数置为0,否则恢复
            if self._clear_history:
                global_step = 0
            else:
                global_step = state_dict['step']
            self.global_step = global_step
            # 恢复权重
            self.model.load_state_dict(state_dict['model'])
            # 计算恢复的轮数
            self.init_eopch = global_step // self._steps_per_epoch
            print('resume at %d steps from %s' % (global_step, self._resume_weights))

    # 准备评估模型的类
    def init_evaluator(self):
        self.evaluator = Evaluator(self.model, self.eval_dataset, self.config)

    # adam优化器
    def init_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self._init_lr)

    def init_losses(self):
        # 4个损失:总损失,位置损失,置信度损失,类别损失
        # 总损失是其余3个之和,各损失均是一段时间的平均损失
        self.losses = {
            'loss': AverageMeter(),
            'giou loss': AverageMeter(),
            'conf loss': AverageMeter(),
            'class loss': AverageMeter(),
        }

    # 评估计算mAP指标
    def eval(self):
        mAP = self.evaluator.evaluate()
        self.APs = mAP.classes
        # 打印每类结果
        for klass in self.APs:
            print('AP for %s = %.4f\n' % (klass, self.APs[klass]))
        self.mAP = mAP.mean
        print('mAP = %.4f\n' % self.mAP)
        return mAP

    def _clear_ap(self):
        self.mAP = None
        self.APs = None

    def save(self, epoch):
        # 如果评估了mAP，则保存轮数和mAP值，否则只保存轮数到文件名
        model_name = f'model-{epoch}.pt' if self.mAP is None else f'model-{epoch}-{self.mAP:.4f}.pt'
        # 保存模型参数
        status = {
            'step': self.global_step,
            'APs': self.APs,
            'mAP': self.mAP,
            'model': self.model.state_dict(),
        }
        torch.save(status, os.path.join(self._weights_dir, model_name))

    def train_epoch(self, epoch):
        # 耗尽一次数据集
        for data in self.train_dataloader:
            self.global_step += 1
            # 将data中的每一个去掉第一个维度(去掉batch_size=1的维度)
            # 依次是:
            #   经过预处理归一化的指定大小的图片(h, w)
            #   格点化的一批图片中小目标的标注(batch_size, h/8, w/8, 3, 6+类别数)
            #   格点化的一批图片中中目标的标注(batch_size, h/16, w/16, 3, 6+类别数)
            #   格点化的一批图片中大目标的标注(batch_size, h/32, w/32, 3, 6+类别数)
            #   原始的一批图片中小目标的标注(batch_size, ?, 4)
            #   原始的一批图片中中目标的标注(batch_size, ?, 4)
            #   原始的一批图片中大目标的标注(batch_size, ?, 4)
            image, label_sbbox, label_mbbox, label_lbbox,\
                sbbox, mbbox, lbbox = [item.squeeze(0) for item in data]

            # 调整学习率
            lr = self.adjust_lr(self.global_step)
            # 清除梯度
            self.optimizer.zero_grad()
            # 前向转播计算总损失和其他部分损失
            loss, *partial_losses = self.model(image, (label_sbbox, label_mbbox, label_lbbox, sbbox, mbbox, lbbox))
            # 反向转播更新参数
            loss.backward()

            # 如果稀疏训练
            if self._sparse_train:
                # 对每个收集的BN module施加L1惩罚
                for m in self.bns:
                    m.weight.grad.data.add_(self._sparse_ratio * torch.sign(m.weight.data))

            self.optimizer.step()

            # 更新每个损失的记录值
            for name, loss_val in zip(self.losses.keys(), (l.item() for l in [loss, *partial_losses])):
                self.losses[name].update(loss_val)

            # 如果到达打印的步数
            if 1 or self.global_step % self._loss_print_interval == 0:
                # 去除每个损失的平均值并重置
                losses = [l.get_avg_reset() for l in self.losses.values()]
                # 依次打印：学习率(lr);当前轮数/最大轮数(epoch);步数(step);训练总平均损失(train_loss)
                # 位置平均损失(xy);置信度平均损失(conf);类别平均损失(cls)
                print('lr: %.6f\tepoch: %d/%d\tstep: %d\ttrain_loss: %.4f(xy: %.4f, conf: %.4f, cls: %.4f)' %
                    (lr, epoch, self._max_epochs, self.global_step, *losses))

        # 如果稀疏训练
        if self._sparse_train:
            # 排序BN层gamma大小，统计20%,40%,60%,80%,100%位置的gamma大小
            # 由此可以看出稀疏化水平
            bn_vals = []
            for m in self.bns:
                bn_vals.extend([v.item() for v in m.weight.data.abs().clone()])
            bn_vals.sort()
            gap = int(len(bn_vals)/5)
            peek = [bn_vals[index] for index in chain(range(gap, len(bn_vals), gap), [-1])]
            print('sparse level: {}'.format(peek))

    def train(self):
        # 每一轮训练
        for epoch in range(self.init_eopch, self._max_epochs):
            self.model.train()
            self._clear_ap()
            self.train_epoch(epoch)
            # 轮数超过了指定轮数
            if epoch >= self._eval_after:
                # 设置模型为评估模式
                self.model.eval()
                self.eval()
                # 重新设置为训练模式
                self.model.train()
            self.save(epoch)

    def run(self):
        ensure_weights_dir(self._weights_dir)
        self.init_dataset()
        # 一轮训练的步数
        self._steps_per_epoch = len(self.train_dataloader)
        # 每一轮训练打印多少次损失
        self._loss_print_interval = self._steps_per_epoch // 10
        self.init_model()
        self.init_weights()
        self.init_evaluator()
        self.init_optimizer()
        self.init_losses()
        if self._sparse_train:
            self.bns = get_bn_modules(self.model)
        self.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='trainer configuration')
    parser.add_argument('--yaml', default='yamls/yolo-lite.yaml')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.yaml)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    Trainer(cfg).run()
