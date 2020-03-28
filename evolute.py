import argparse
import json
import time
from copy import deepcopy

import numpy as np

import tools
from config import cfg
from model.parser import YOLOLayer
from trainer import Trainer


def p_gen(x):
    return np.random.beta(1.5, 1.5)

def norm_gen(x):
    return x * float((np.random.randn(1) * 0.2 + 1) ** 2.0)

class Evoluter(Trainer):

    def __init__(self, config):
        super().__init__(config)
        np.random.seed(int(time.time()))

        self.scheduler = self.scheduler_step
        self._warmup_epochs = 0

        self.yolos = []
        self.records = []

        self.hypers = {
            'hflip_p': 0.5,
            'crop_p': 0.75,
            'color_p': 0.2,
            'mixup_p': 0.5,
            'ignore_thresh': 0.5,
            'bbox_loss_gain': 1,
            'conf_loss_gain': 1,
            'cls_loss_gain': 1.7,
            'conf_loss_alpha': 0.5,
            'cls_loss_alpha': 0.5,
            'conf_loss_beta': 2,
            'cls_loss_beta': 2,
        }
        self.hyper_gener = {
            'hflip_p': p_gen,
            'crop_p': p_gen,
            'color_p': p_gen,
            'mixup_p': p_gen,
            'ignore_thresh': p_gen,
            'bbox_loss_gain': norm_gen,
            'conf_loss_gain': norm_gen,
            'cls_loss_gain': norm_gen,
            'conf_loss_alpha': p_gen,
            'cls_loss_alpha': p_gen,
            'conf_loss_beta': None,
            'cls_loss_beta': None,
        }

    def fit(self):
        self.init_dataset()
        self._steps_per_epoch = len(self.train_dataloader)
        self._loss_print_interval = self._steps_per_epoch // 10
        self.init_evaluator()
        self.init_optimizer()
        self.init_losses()

        self.model.train()
        self._clear_ap()

        self.epoch_tt.tic()
        for i in range(1):
            self.train_epoch(i)
        self.epoch_tt.toc()
        print('{:.3f}s'.format(self.epoch_tt.sum_reset()/1e9))

        # 设置模型为评估模式
        self.model.eval()
        self.eval()

        del self.train_dataloader
        del self.eval_dataloader
        del self.evaluator
        del self.optimizer
        return self.mAP

    def random_hyper(self):
        hypers = {}
        for k in self.hypers:
            if k == 'conf_loss_beta':
                hypers[k] = 2
            elif k == 'cls_loss_beta':
                hypers[k] = int(np.random.choice(range(3)))
            else:
                hypers[k] = self.hyper_gener[k](self.hypers[k])

        self.config.augment.color_p = hypers['color_p']
        self.config.augment.mixup_p = hypers['mixup_p']
        self.config.augment.hflip_p = hypers['hflip_p']
        self.config.augment.crop_p = hypers['crop_p']

        for layer in self.yolos:
            opt = layer.opt
            opt['ignore_thresh'] = hypers['ignore_thresh']
            opt['bbox_loss_gain'] = hypers['bbox_loss_gain']
            opt['conf_loss_gain'] = hypers['conf_loss_gain']
            opt['cls_loss_gain'] = hypers['cls_loss_gain']
            opt['conf_loss_alpha'] = hypers['conf_loss_alpha']
            opt['cls_loss_alpha'] = hypers['cls_loss_alpha']
            opt['conf_loss_beta'] = hypers['conf_loss_beta']
            opt['cls_loss_beta'] = hypers['cls_loss_beta']
        return hypers

    def run(self):
        tools.ensure_dir(self._weights_dir)

        self.init_cfg()
        self.init_dataset()
        self._steps_per_epoch = len(self.train_dataloader)
        self._loss_print_interval = self._steps_per_epoch // 10
        self.init_model()

        for layer in self.model.module.module_list:
            if isinstance(layer, YOLOLayer):
                self.yolos.append(layer)

        state_dict = deepcopy(self.model.state_dict())
        for i in range(200):
            self.model.load_state_dict(state_dict)
            hypers = self.random_hyper()
            print(i, hypers)
            fitness = self.fit()
            print(fitness)
            self.records.append({
                'hyper': hypers,
                'fitness': fitness,
            })
            json.dump({'data': self.records}, open('evol2.json', 'w'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evoluter configuration')
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
    # cfg.freeze()
    print(cfg)
    Evoluter(cfg).run()
