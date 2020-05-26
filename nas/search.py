import json
from os import path

import numpy as np
import torch
from thop import clever_format, profile
from torch import nn

import tools
from config import cfg
from nas.detnet import detnet_600m
from trainer import Trainer
from typing import Optional


class JSONSaver:

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.records = []
        if path.exists(save_path):
            self.records = json.load(open(save_path, 'r'))['data']

    def append(self, record):
        self.records.append(record)
        self.save()

    def save(self):
        json.dump({'data': self.records}, open(self.save_path, 'w'), cls=NpEncoder)

class NpEncoder(json.JSONEncoder):

    # pylint: disable-msg=method-hidden
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def generate_model(macs_thres=15e9, time_thres=(0, 100), gen_func=detnet_600m):
    while True:
        net = gen_func().cuda()
        inputs = torch.randn(1, 3, 512, 512).cuda()
        flops, params = profile(net, inputs=(inputs, ), verbose=False)
        if flops > macs_thres:
            continue
        avg_time = tools.compute_time(net, batch_size=16)
        if avg_time > time_thres[1] or avg_time < time_thres[0]:
            continue
        net.attr = {
            'MACs': flops,
            'params': params,
            'avg_time': avg_time,
        }
        print(net.cfg)
        flops, params = clever_format([flops, params], "%.3f")
        print('MACs: {}, params: {}, {:.2f} ms'.format(flops, params, avg_time))
        yield nn.DataParallel(net)

def train_model(model: nn.Module):
    trainer = Trainer(cfg)
    mAP = trainer.run_nas(model)
    del trainer
    return mAP

def _record_train_model(js: Optional[JSONSaver], model: nn.Module, add_record: bool=False):
    ok = False
    try:
        mAP = train_model(model)
        print(mAP)
    except Exception as e:
        print('*** TRAIN ERROR ***')
        print(e)
    else:
        ok = True
        if add_record:
            _add_model_record(js, model, mAP)
    del model
    return ok

def _merge_cfg(cfg_path: str='yamls/nas.yaml'):
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    print(cfg)

def _add_model_record(js: JSONSaver, model: nn.Module, mAP: float):
    model = tools.bare_model(model)
    record = {
        'cfg': model.cfg,
        'mAP': mAP,
    }
    record.update(model.attr)
    js.append(record)

def search(json_path: str, num: int=500):
    _merge_cfg()
    js = JSONSaver(json_path)

    i = 0
    for model in generate_model(time_thres=[45, 65]):
        print(i + 1, '*' * 20)
        if not _record_train_model(js, model, add_record=True):
            continue
        i += 1
        if i >= num:
            break

def order_model(json_path: str, fpn_cfg, add_record=True):
    _merge_cfg()
    js = JSONSaver(json_path)
    for model in generate_model(9e99, [0, 9e99], lambda: detnet_600m(fpn_cfg)):
        _record_train_model(js, model, add_record=add_record)
        return

if __name__ == "__main__":
    # FPN=yolo-lite, mAP=0.5812078947769721
    jp = 'results/nas2.json'
    search(jp, 500)
    # order_model(
    #     jp,
    #     [{'w_in': 528, 'w_out': 280, 'd': 3, 'p': 2, 'bm': 1.0, 'gw': 40, 'stride': 32, 'merge': 0}, {'w_in': 280, 'w_out': 480, 'd': 3, 'p': 2, 'bm': 1.0, 'gw': 40, 'stride': 16, 'merge': 240}, {'w_in': 480, 'w_out': 120, 'd': 3, 'p': 2, 'bm': 1.0, 'gw': 40, 'stride': 8, 'merge': 96}]
    # )
