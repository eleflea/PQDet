import torch
from torch import nn
import numpy as np
from eval.evaluate import Evaluator
from itertools import chain
from collections import OrderedDict
from pruning.block import CB, Conv
import config as cfg

from model.backbone.mobilenetv2 import InvertedResidual


def make_model_weights(model_cls, weights_path):
    model = model_cls().cuda().eval()
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict['model'])
    # print('load weights from %s' % weights_path)
    return model

class BasePruner:
    def __init__(self, model_cls):
        self.model = make_model_weights(model_cls, cfg.PRUNE_WEIGHTS)
        self.new_model = make_model_weights(model_cls, cfg.PRUNE_WEIGHTS)
        self.blocks = []
        self.prune_ratio = cfg.PRUNE_RATIO

    def prune(self):
        blocks = OrderedDict()
        previous_module = None
        previous_name = ''
        last_block = None

        # set InvertedResidual residual connect
        for mod in self.new_model.modules():
            if not isinstance(mod, InvertedResidual):
                continue
            if mod.use_res_connect:
                mod.conv[2].keep_output = True

        for _, (name, module) in enumerate(chain(self.new_model.named_modules(), [['pad', None]])):
            if not isinstance(previous_module, nn.Conv2d):
                previous_module = module
                previous_name = name
                continue
            idx = len(blocks)
            if isinstance(module, nn.BatchNorm2d):
                block = CB(
                    previous_name,
                    idx,
                    [last_block],
                    [previous_module, module],
                    [*list(previous_module.state_dict().values()), *list(module.state_dict().values())],
                    keep_output=hasattr(previous_module, 'keep_output'),
                )
            else:
                block = Conv(
                    previous_name,
                    idx,
                    [last_block],
                    [previous_module],
                    list(previous_module.state_dict().values()),
                )
            blocks[previous_name] = block
            last_block = block
            previous_module = module
            previous_name = name
        blocks['mergelarge.conv7.convbn.conv'].input_layer = [blocks['headslarge.conv4.convbn.conv']]
        blocks['headsmid.conv8.convbn.conv'].input_layer.append(blocks['backbone.features.13.conv.2'])
        blocks['mergemid.conv15.convbn.conv'].input_layer = [blocks['headsmid.conv12.convbn.conv']]
        blocks['headsmall.conv16.convbn.conv'].input_layer.append(blocks['backbone.features.6.conv.2'])
        self.blocks = blocks
    def test(self):
        evaluator = Evaluator(self.new_model)
        self.new_model.eval()
        APs = evaluator.mAP()
        for cls in APs:
            AP_mess = 'AP for %s = %.4f\n' % (cls, APs[cls])
            print(AP_mess.strip())
        mAP = np.mean([APs[cls] for cls in APs])
        mAP_mess = 'mAP = %.4f\n' % mAP
        print(mAP_mess.strip())
    def finetune(self, epoch=10):
        dataset = YOLODataset()
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True,
        )
        cfg.STEPS_PER_EPOCH = len(dataloader)
        self.trainer.model=self.newmodel
        # self.best_mAP=self.trainer._valid_epoch(validiter=10)[0][0]
        self.best_mAP=0
        for epoch in range(0, self.trainer.args.OPTIM.total_epoch):
            self.trainer.global_epoch += 1
            self.trainer._train_epoch()
            self.trainer.lr_scheduler.step(epoch)
            lr_current = self.trainer.optimizer.param_groups[0]['lr']
            print("epoch:{} lr:{}".format(epoch,lr_current))
            results, imgs = self.trainer._valid_epoch()
            self.trainer._reset_loggers()
            if results[0] > self.best_mAP:
                self.best_mAP = results[0]
                self.trainer._save_ckpt(name='best-ft{}'.format(self.pruneratio), metric=self.best_mAP)
        return self.best_mAP
