import torch
from torch import nn
from torch.quantization import DeQuantStub, QuantStub

from model.parser import Parser
from typing import IO, Union


def item_getter(*items):
    def getter(x):
        if x is None:
            return None
        return tuple(x[i] for i in items)
    return getter

_TARGET_MAP = {
    8: item_getter(0, 3),
    16: item_getter(1, 4),
    32: item_getter(2, 5),
}

class AnyModel(nn.Module):

    def __init__(self, cfg: Union[str, IO], quant: bool=False, onnx: bool=False):
        super().__init__()
        self.quant = quant
        self.qstub = QuantStub()
        self.destub = DeQuantStub()

        if isinstance(cfg, str):
            cfg = open(cfg, 'r')
        self.module_list = nn.ModuleList(Parser(cfg).torch_layers(quant, onnx))
        cfg.close()

    def is_output(self, i, layer) -> bool:
        return False

    def forward(self, x, target=None):
        cache_outputs = []
        outputs = []
        for i, layer in enumerate(self.module_list):
            if self.quant and i == 0:
                x = self.qstub(x)
            layer_type = layer._type
            if layer_type in ('convolutional', 'fc', 'upsample', 'maxpool', 'avgpool'):
                x = layer(x)
            elif layer_type in {'shortcut', 'scale_channels'}:
                x = layer(x, cache_outputs[layer._from])
            elif layer_type == 'route':
                x = layer([cache_outputs[li] for li in layer._layers])
            elif layer_type == 'yolo':
                if self.quant:
                    x = self.destub(x)
                x = layer(x, _TARGET_MAP[layer._stride](target))
            else:
                raise ValueError('unknown layer type: %s' % layer_type)
            if self.is_output(i, layer):
                outputs.append(x)
            cache_outputs.append(x)
        num_outputs = len(outputs)
        if num_outputs == 0:
            outputs = cache_outputs[-1]
        elif num_outputs == 1:
            outputs = outputs[0]
        return outputs

class DetectionModel(AnyModel):

    def is_output(self, i, layer) -> bool:
        return layer._type == 'yolo'

    def forward(self, x, target=None):
        outputs = super(DetectionModel, self).forward(x, target)
        if target is None:
            outputs = [output.view((output.shape[0], -1, output.shape[-1])) for output in outputs]
            return torch.cat(outputs, dim=1)
        losses = list(map(sum, zip(*outputs)))
        loss_per_branch = [sum(loss[1:]) for loss in outputs]
        return {
            'loss': losses[0],
            'giou_loss': losses[1],
            'conf_loss': losses[2],
            'class_loss': losses[3],
            'loss_per_branch': loss_per_branch,
        }

class ClassifierModel(AnyModel):
    pass

if __name__ == "__main__":
    Model = DetectionModel
    # print(Model('model/cfg/mobilenetv2-yolo.cfg').module_list)
    from thop import clever_format, profile
    model = Model('model/cfg/regnety-400m-fpn.cfg')
    inputs = torch.randn(1, 3, 512, 512)
    flops, params = profile(model, inputs=(inputs, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("flops:{}, params: {}".format(flops, params))
