import argparse
import math
import os
import numpy as np
from itertools import chain
from copy import deepcopy

import torch
import torch.optim as optim
from torch import nn

from config import cfg, get_device, fix_gpus
from dataset.eval_dataset import EvalDataset
from dataset.train_dataset import TrainDataset, collate_batch
from eval.evaluator import Evaluator
from model.newyolo import YOLOv3
from tools import AverageMeter, TicToc


def get_bn_modules(model: nn.DataParallel):
    '''遍历整个模型，得到所有需要稀疏化的BN module，暂时是全部。

    Args:
        model: 网络模型。

    Returns:
        list: 所有需要稀疏化的BN module的列表。
    '''
    all_bns = []
    c = 0
    for l in model.module.module_list:
        if l._type == 'convolutional' and hasattr(l, 'bn'):
            if not hasattr(l, '_notprune'):
                all_bns.append(l.bn)
            c += 1
    print("sparse mode: {}/{} BN layers will be sparsed.".format(len(all_bns), c))
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
        self._scheduler_type = config.train.scheduler
        self._mile_stones = config.train.mile_stones
        self._gamma = config.train.gamma
        self._init_lr = config.train.learning_rate_init
        self._end_lr = config.train.learning_rate_end
        self._warmup_epochs = config.train.warmup_epochs
        self._max_epochs = config.train.max_epochs
        # weights
        self._backbone_weight = config.weight.backbone
        self._weights_dir = os.path.join(config.weight.dir, config.experiment_name)
        self._resume_weight = config.weight.resume
        self._clear_history = config.weight.clear_history
        self._weight_base_name = 'model'
        # eval
        self._eval_after = config.eval.after
        # sparse
        self._sparse_train = config.sparse.switch
        self._sparse_ratio = config.sparse.ratio
        # quant
        self._quant_train = config.quant.switch
        self._disable_observer_after = config.quant.disable_observer_after
        self._freeze_bn_after = config.quant.freeze_bn_after
        # system
        self._gpus = fix_gpus(config.system.gpus)
        self._num_workers = config.system.num_workers
        self._device = get_device(self._gpus)

        self.init_eopch = 0
        self.global_step = 0
        self.config = config

        self.dataload_tt = TicToc()
        self.model_tt = TicToc()
        self.epoch_tt = TicToc()

        self.scheduler = {
            'cosine': self.scheduler_cosine,
            'step': self.scheduler_step,
        }[self._scheduler_type]

    def scheduler_cosine(self, steps: int):
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

    def scheduler_step(self, steps: int):
        '''根据训练步数通过公式计算多步的学习率,
        通过对优化器中每个参数赋值调整学习率。

        Args:
            steps: 当前训练的步数。

        Returns:
            float: 学习率。
        '''
        # 热身步数
        warmup_steps = self._warmup_epochs * self._steps_per_epoch
        if steps < warmup_steps:
            lr = steps / warmup_steps * self._init_lr
        else:
            for i, m in enumerate(chain(self._mile_stones, [self._max_epochs])):
                if steps < m * self._steps_per_epoch:
                    lr = self._init_lr * self._gamma ** i
                    break
        # 对每个参数赋值新的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def fuse_model(self):
        for layer in self.model.module.module_list:
            if layer._type == 'convolutional':
                names = [name for name, _ in layer.named_children() if name in {'conv', 'bn', 'act'}]
                if 'bn' not in names:
                    continue
                torch.quantization.fuse_modules(layer, names, inplace=True)

    def init_quant(self):
        print('quantization aware training')
        self.fuse_model()
        self.model = self.model.module
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self.model, inplace=True)
        self.model.to(self._device)

    # 建立数据集
    def init_dataset(self):
        train_dataset = TrainDataset(self.config)
        eval_dataset = EvalDataset(self.config)
        # 数据集内部手动生成batch,所以此处batch_size=None
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self._train_batch_size,
            shuffle=False, num_workers=self._num_workers,
            pin_memory=True, collate_fn=collate_batch,
        )
        self.eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=None, shuffle=False,
            num_workers=self._num_workers, pin_memory=True,
            collate_fn=lambda x: x,
        )
        print(f'{train_dataset.length} images for train.')
        print(f'{eval_dataset.length} images for evaluate.')

    # 建立YOLOv3模型
    def init_model(self):
        model = YOLOv3(self._cfg_path, self._quant_train).to(self._device)
        model = nn.DataParallel(model, device_ids=self._gpus)
        self.model = model

    def init_weights(self):
        # 如果backbone权重存在，载入做迁移学习
        if len(self._backbone_weight) > 0:
            print('loading backbone weights from {}'.format(self._backbone_weight))
            state_dict = torch.load(self._backbone_weight, map_location=self._device)
            ori_state_dict = self.model.module.state_dict()
            ori_state_dict.update(state_dict)
            self.model.module.load_state_dict(ori_state_dict)

        # 如果是恢复训练进度
        if len(self._resume_weight) > 0:
            # 读入待恢复的权重和训练步数
            state_dict = torch.load(self._resume_weight, map_location=self._device)
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
            print('resume at %d steps from %s' % (global_step, self._resume_weight))

    # 准备评估模型的类
    def init_evaluator(self):
        self.evaluator = Evaluator(self.model, self.eval_dataloader, self.config)

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
            print('AP@%s = %.4f' % (klass, self.APs[klass]))
        self.mAP = mAP.mean
        print('mAP = %.4f' % self.mAP)
        return mAP

    def _clear_ap(self):
        self.mAP = None
        self.APs = None

    def save(self, epoch, jit=False):
        # 如果评估了mAP，则保存轮数和mAP值，否则只保存轮数到文件名
        base_name = self._weight_base_name
        if jit:
            base_name = 'jit-' + base_name
        model_name = f'{base_name}-{epoch}.pt' if self.mAP is None\
            else f'{base_name}-{epoch}-{self.mAP:.4f}.pt'
        model_path = os.path.join(self._weights_dir, model_name)
        if jit:
            model = torch.quantization.convert(self.model.to('cpu').eval(), inplace=False)
            torch.jit.save(torch.jit.script(model.eval()), model_path)
        # 保存模型参数
        status = {
            'step': self.global_step,
            'APs': self.APs,
            'mAP': self.mAP,
            'model': self.model.state_dict(),
        }
        torch.save(status, model_path)

    def train_epoch(self, epoch):
        # 耗尽一次数据集
        self.dataload_tt.tic()
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
            if self._quant_train:
                data = [item.cuda() for item in data]
            image, label_sbbox, label_mbbox, label_lbbox,\
                sbbox, mbbox, lbbox = data
            self.dataload_tt.toc()

            # 调整学习率
            lr = self.scheduler(self.global_step)

            self.model_tt.tic()
            # 前向转播计算总损失和其他部分损失
            loss, *partial_losses = self.model(image, (label_sbbox, label_mbbox, label_lbbox, sbbox, mbbox, lbbox))
            # 清除梯度
            self.optimizer.zero_grad()
            # 反向转播更新参数
            loss.mean().backward()

            # 如果稀疏训练
            if self._sparse_train:
                # 对每个收集的BN module施加L1惩罚
                for m in self.bns:
                    m.weight.grad.data.add_(self._sparse_ratio * torch.sign(m.weight.data))

            self.optimizer.step()
            self.model_tt.toc()

            # 更新每个损失的记录值
            for name, loss_val in zip(self.losses.keys(), (l.mean().item() for l in [loss, *partial_losses])):
                self.losses[name].update(loss_val)

            # 如果到达打印的步数
            if self.global_step % self._loss_print_interval == 0:
                # 去除每个损失的平均值并重置
                losses = [l.get_avg_reset() for l in self.losses.values()]
                # 依次打印：学习率(lr);当前轮数/最大轮数(epoch);步数(step);训练总平均损失(train_loss)
                # 位置平均损失(xy);置信度平均损失(conf);类别平均损失(cls)
                print('lr: %.6f\tepoch: %d/%d\tstep: %d\ttrain_loss: %.4f(xy: %.4f, conf: %.4f, cls: %.4f)' %
                    (lr, epoch, self._max_epochs, self.global_step, *losses))

            self.dataload_tt.tic()

        self.train_dataloader.dataset.init_shuffle()

        # 如果稀疏训练
        if self._sparse_train:
            # 排序BN层gamma大小，统计20%,40%,60%,80%,100%位置的gamma大小
            # 由此可以看出稀疏化水平
            bn_vals = np.concatenate([m.weight.data.abs().clone().cpu().numpy() for m in self.bns])
            bn_vals.sort()
            bn_num = len(bn_vals)
            bn_indexes = [round(i/5*bn_num)-1 for i in range(1, 6)]
            print('sparse level: {}'.format(bn_vals[bn_indexes].tolist()))

        print('data load time: {:.3f}s, model train time: {:.3f}s'.format(
            self.dataload_tt.sum_reset()/1e9, self.model_tt.sum_reset()/1e9
        ))

    def train(self):
        # 每一轮训练
        for epoch in range(self.init_eopch, self._max_epochs):
            self.model.train()
            self._clear_ap()

            self.epoch_tt.tic()
            self.train_epoch(epoch)
            self.epoch_tt.toc()
            print('{:.3f}s per epoch'.format(self.epoch_tt.sum_reset()/1e9))

            if self._quant_train:
                if epoch >= self._disable_observer_after:
                    # Freeze quantizer parameters
                    self.model.apply(torch.quantization.disable_observer)
                if epoch >= self._freeze_bn_after:
                    # Freeze batch norm mean and variance estimates
                    self.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

            # 轮数超过了指定轮数
            if epoch >= self._eval_after:
                if self._quant_train:
                    quant_model = deepcopy(self.model)
                    self.evaluator.model = torch.quantization.convert(
                        quant_model.eval().cpu(), inplace=False
                    )
                    self.evaluator.model.eval()
                    self.eval()
                    self.save(epoch)
                    continue
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
        if self._quant_train:
            self.init_quant()
        self.init_losses()
        if self._sparse_train:
            self.bns = get_bn_modules(self.model)
        self.train()

    def run_prune(self, prune_weight: str):
        self._cfg_path = self.config.prune.new_cfg
        self._init_lr *= 0.2
        self._warmup_epochs = 0
        self._max_epochs = 20
        self._backbone_weight = ''
        self._resume_weight = prune_weight
        self._clear_history = True
        self._eval_after = 0
        self._sparse_train = False
        self._weight_base_name = 'pruned-model'
        self.run()

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
    print(cfg)
    Trainer(cfg).run()
