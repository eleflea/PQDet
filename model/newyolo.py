import torch
from torch import nn

from model.parser import Parser


def item_getter(*items):
    def getter(x):
        if x is None:
            return None
        return tuple(x[i] for i in items)
    return getter

class YOLOv3(nn.Module):

    def __init__(self, cfg_path: str):
        super().__init__()

        with open(cfg_path, 'r') as fr:
            self.module_list = nn.ModuleList(Parser(fr).torch_layers())

    def forward(self, x, target=None):
        cache_outputs = []
        outputs = []
        target_map = {
            8: item_getter(0, 3),
            16: item_getter(1, 4),
            32: item_getter(2, 5),
        }
        for layer in self.module_list:
            layer_type = layer._type
            if layer_type in {'convolutional', 'upsample', 'maxpool'}:
                x = layer(x)
            elif layer_type == 'shortcut':
                x += cache_outputs[layer._from]
            elif layer_type == 'route':
                x = torch.cat([cache_outputs[li] for li in layer._layers], dim=1)
            elif layer_type == 'yolo':
                x = layer(x, target_map[layer.stride](target))
                outputs.append(x)
            else:
                raise ValueError('unknown layer type: %s' % layer_type)
            cache_outputs.append(x)

        if target is None:
            outputs = [output.view((output.shape[0], -1, output.shape[-1])) for output in outputs]
            return torch.cat(outputs, dim=1)
        losses = list(map(lambda *x: sum(x), *outputs))
        return losses

if __name__ == "__main__":
    # print(YOLOv3('model/cfg/mobilenetv2-yolo.cfg').module_list)
    from thop import clever_format, profile
    # from model.yolov3 import YOLOv3
    model = YOLOv3('model/cfg/mobilenetv2-yolo.cfg')
    inputs = torch.randn(1, 3, 416, 416)
    flops, params = profile(model, inputs=(inputs, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("flops:{}, params: {}".format(flops, params))
