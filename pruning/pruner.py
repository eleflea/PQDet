from copy import deepcopy

import torch
from torch import nn

from dataset.eval_dataset import EvalDataset
from eval.evaluator import Evaluator
from pruning import block as PB
from pruning.block import Conv2d
from trainer import Trainer
from config import get_device


CFG_NET_SEGMENT = '''[net]
# Testing
#batch=1
#subdivisions=1
# Training
batch=16
subdivisions=1
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1'''

class SlimmingPruner:
    def __init__(self, model_fun, cfg):
        self.cfg = cfg
        self._prune_weight = cfg.prune.weight
        self._prune_ratio = cfg.prune.ratio
        self._new_cfg = cfg.prune.new_cfg
        self._num_workers = cfg.system.num_workers
        self._device = get_device(cfg.system.gpus)
        state_dict = torch.load(self._prune_weight, map_location=self._device)
        model = model_fun()
        new_model = model_fun()
        model.load_state_dict(state_dict['model'])
        new_model.load_state_dict(state_dict['model'])
        print('load weights from %s' % self._prune_weight)
        self.model = model
        self.new_model = new_model
        self.blocks = []
        self._pruned_weight = self._prune_weight.rsplit('.', 1)[0] + '-pruned.pt'

        self.block_map = {
            'convolutional': PB.Conv2d,
            'maxpool': PB.Pool,
            'upsample': PB.Upsample,
            'yolo': PB.YOLO,
            'shortcut': PB.ShortCut,
            'route': PB.Route
        }

    def prune(self):
        for i, layer in enumerate(self.new_model.module.module_list):
            if layer._type in {'convolutional', 'maxpool', 'upsample', 'yolo'}:
                input_layers = [] if len(self.blocks) == 0 else [self.blocks[-1]]
            elif layer._type == 'shortcut':
                self.blocks[layer._from].keep_out = True
                self.blocks[-1].keep_out = True
                input_layers = [self.blocks[layer._from], self.blocks[-1]]
            elif layer._type == 'route':
                input_layers = [self.blocks[li] for li in layer._layers]
            else:
                raise ValueError("unknown layer type '%s'" % layer._type)
            self.blocks.append(self.block_map[layer._type](i, input_layers, layer))

        # gather BN weights
        bns = []
        maxbn = []
        for b in self.blocks:
            if isinstance(b, Conv2d) and b.bn_scale is not None:
                bns.extend(b.bn_scale.tolist())
                maxbn.append(b.bn_scale.max().item())

        bns = torch.Tensor(bns)
        sorted_bns = torch.sort(bns)[0]
        prune_limit = (sorted_bns == min(maxbn)).nonzero().item() / len(bns)
        print('prune limit: {}'.format(prune_limit))
        if self._prune_ratio > prune_limit:
            raise AssertionError('prune ratio bigger than limit')

        thre_index = int(bns.shape[0] * self._prune_ratio)
        thre = sorted_bns[thre_index]
        thre = thre.to(self._device)
        pruned_bn = 0
        segments = [CFG_NET_SEGMENT]
        for b in self.blocks:
            pruned_num = b.prune(thre)
            pruned_bn += pruned_num
            print("({}){}: {}/{} pruned".format(
                b.layer_id, b.layer_name, pruned_num, len(b.out_mask)
            ))
            segments.append(b.reflect())
        cfg_content = '\n\n'.join(segments)
        with open(self._new_cfg, 'w') as fw:
            fw.write(cfg_content)

        status = {
            'step': 0,
            'model': self.new_model.state_dict(),
        }
        torch.save(status, self._pruned_weight)
        print("Slimming Pruner done")

    def test(self):
        eval_dataset = EvalDataset(self.cfg)
        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=None, shuffle=False,
            num_workers=self._num_workers, pin_memory=True,
            collate_fn=lambda x: x,
        )
        evaluator = Evaluator(self.new_model, dataloader, self.cfg)
        self.new_model.eval()
        AP = evaluator.evaluate()
        APs = AP.classes
        # 打印每类结果
        for klass in APs:
            print('AP@%s = %.4f' % (klass, APs[klass]))
        print('mAP = %.4f' % AP.mean)

    def finetune(self):
        trainer = Trainer(self.cfg)
        trainer.run_prune(self._pruned_weight)
