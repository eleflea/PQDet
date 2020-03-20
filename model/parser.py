'''
parse a darnet cfg file to pytorch network
'''

from collections import namedtuple
from typing import IO, Generator, List, Tuple, Union

import torch
from torch import nn
from torch.nn.quantized import FloatFunctional

from model.loss import loss_per_scale

Layer = namedtuple('Layer', ['name'])
Attr = namedtuple('Attr', ['attr', 'val'])

DEFAULT_LAYERS = {
    'net': {
        'name': 'net',
        'channels': 3,
    },
    'convolutional': {
        'name': 'convolutional',
        'filters': 1,
        'size': 1,
        'stride': 1,
        'pad': 0,
        'padding': 0,
        'groups': 1,
        'activation': 'logistic',
        'batch_normalize': 0,
    },
    'shortcut': {
        'name': 'shortcut',
        'activation': 'linear',
        'alpha': 1, # not use
        'beta': 1, # not use
    },
    'route': {
        'name': 'route',
        'layers': -1,
    },
    'maxpool': {
        'name': 'maxpool',
        'size': 1,
        'stride': 1,
        'pad': 0,
        'padding': 0,
    },
    'upsample': {
        'name': 'upsample',
        'stride': 2,
    },
    'yolo': {
        'name': 'yolo',
        'classes': 1,
        'ignore_thresh': .5,
        'bbox_loss': 'giou',
        'l1_loss_gain': 0.1,
    }
}

ACTIVATION_MAP = {
    'logistic': nn.Sigmoid,
    'leaky': lambda: nn.LeakyReLU(0.1, inplace=True),
    'relu': lambda: nn.ReLU(inplace=True),
    'relu6': lambda: nn.ReLU6(inplace=True),
    'tanh': nn.Tanh,
}

def str2value(ns):
    try:
        if '.' not in ns:
            return int(ns)
        return float(ns)
    except ValueError:
        return ns

class ShortCut(nn.Module):
    def __init__(self, activation: str, quant: bool=False):
        super().__init__()
        self.quant = quant
        self.ffunc = FloatFunctional()
        self.act = None
        if activation != 'linear':
            self.act = ACTIVATION_MAP[activation]()

    def forward(self, x, other):
        if self.act is not None:
            x = self.act(x)
        if self.quant:
            return self.ffunc.add(x, other)
        x += other
        return x

class Route(nn.Module):
    def __init__(self, quant: bool=False):
        super().__init__()
        self.quant = quant
        self.ffunc = FloatFunctional()

    def forward(self, xs):
        if self.quant:
            return self.ffunc.cat(xs, dim=1)
        return torch.cat(xs, dim=1)

class Decode(nn.Module):
    def __init__(self, num_classes: int, stride: int):
        super(Decode, self).__init__()
        self.num_classes = num_classes
        self.stride = stride
        self.grid_size = (0, 0)

    def respawn_grid(self, height, width, device):
        shiftx = torch.arange(0, height, dtype=torch.float32) + 0.5
        shifty = torch.arange(0, width, dtype=torch.float32) + 0.5
        shifty, shiftx = torch.meshgrid(shiftx, shifty)
        shiftx = shiftx.unsqueeze(-1)
        shifty = shifty.unsqueeze(-1)
        self.xy_grid = torch.stack([shiftx, shifty], dim=-1).to(device)
        self.grid_size = (height, width)

    def forward(self, conv):
        conv = conv.permute(0, 2, 3, 1)
        conv_shape = conv.shape
        batch_size = conv_shape[0]
        out_size_h = conv_shape[1]
        out_size_w = conv_shape[2]
        gt_per_grid = conv_shape[3] // (5 + self.num_classes)

        conv = conv.view(batch_size, out_size_h, out_size_w,
            gt_per_grid, 5 + self.num_classes)
        conv_raw_dx1dy1, conv_raw_dx2dy2, conv_raw_conf, conv_raw_prob = torch.split(
            conv, [2, 2, 1, self.num_classes], dim=-1)

        if out_size_h > self.grid_size[0] or out_size_w > self.grid_size[1]:
            self.respawn_grid(round(out_size_h*1.2), round(out_size_w*1.2), conv.device)
        xy_grid = self.xy_grid[:out_size_h, :out_size_w, ...].to(conv.device)

        # decode xy
        pred_xymin = (xy_grid - torch.exp(conv_raw_dx1dy1)) * self.stride
        pred_xymax = (xy_grid + torch.exp(conv_raw_dx2dy2)) * self.stride
        # decode confidence
        pred_conf = torch.sigmoid(conv_raw_conf)
        # decode probability
        pred_prob = torch.sigmoid(conv_raw_prob)
        # re-concat decoded items
        pred_bbox = torch.cat((pred_xymin, pred_xymax, pred_conf, pred_prob), -1)
        return pred_bbox

class YOLOLayer(nn.Module):

    def __init__(self, opt: dict):
        super().__init__()
        self.decode = Decode(opt['classes'], opt['stride'])
        self.opt = opt

    def forward(self, x, target=None):
        x = self.decode(x)
        if target is None:
            return x
        losses = loss_per_scale(x, *target, self.opt)
        return losses

UnionLA = Union[Layer, Attr]

class Parser():

    def __init__(self, fp: IO):
        self.fp = fp
        self.size = 0
        self.pos = -1
        self.peek = ' '
        self.line = None

    def read(self):
        if self.pos == self.size - 1:
            raise EOFError
        self.pos += 1
        self.peek = self.line[self.pos]

    def omit(self):
        while self.peek.isspace():
            self.read()

    def match(self, c):
        if self.peek == c:
            self.read()
        else:
            raise SyntaxError("expect '{}', got '{}'".format(c, self.peek))

    def name(self) -> str:
        temp = []
        i = 0
        while (self.peek.isalpha() or self.peek == '_') or (i != 0 and self.peek.isdigit()):
            temp.append(self.peek)
            try:
                self.read()
            except EOFError:
                break
            i += 1
        return ''.join(temp)

    def val(self) -> Union[List, str, int, float]:
        vals = []
        temp = []
        while self.peek != '#':
            try:
                if self.peek == ',':
                    ns = ''.join(temp)
                    vals.append(str2value(ns))
                    temp.clear()
                    self.read()
                    continue
                temp.append(self.peek)
                self.read()
            except EOFError:
                break
        if len(vals) == 0:
            return str2value(''.join(temp))
        if len(temp) != 0:
            vals.append(str2value(''.join(temp)))
        return vals

    def parse_line(self) -> Union[Layer, Attr]:
        self.read()
        if self.peek == '[':
            self.read()
            layer_name = self.name()
            return Layer(layer_name)
        else:
            attr_name = self.name()
            self.omit()
            self.match('=')
            self.omit()
            attr_val = self.val()
            return Attr(attr_name, attr_val)

    def gen_lines(self) -> Generator[Tuple[int, str], None, None]:
        for i, line in enumerate(self.fp, 1):
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            yield (i, line)

    def gen_blocks(self) -> Generator[UnionLA, None, None]:
        for i, line in self.gen_lines():
            self.line = line
            self.size = len(line)
            self.pos = -1
            try:
                yield self.parse_line()
            except EOFError:
                raise SyntaxError('line %d: something is missing' % i)
            except SyntaxError as e:
                raise SyntaxError(('line %d: ' % i) + str(e))

    def gen_layers(self) -> Generator[dict, None, None]:
        current_layer = None
        for block in self.gen_blocks():
            if isinstance(block, Layer):
                if current_layer is not None:
                    yield current_layer
                current_layer = DEFAULT_LAYERS[block.name].copy()
            elif isinstance(block, Attr):
                if current_layer is not None:
                    # pylint: disable-msg=unsupported-assignment-operation
                    current_layer[block.attr] = block.val
        if current_layer is not None:
            yield current_layer

    def torch_layers(self, quant: bool=False) -> List[nn.Module]:
        input_channels = None
        stride = 1
        layers = []
        for l in self.gen_layers():
            name = l['name']
            if name == 'net':
                input_channels = l['channels']
                continue
            if name == 'convolutional':
                blocks = nn.Sequential()
                padding = l['size'] // 2 if l['pad'] == 1 else l['padding']
                bias = l['batch_normalize'] == 0
                blocks.add_module('conv', nn.Conv2d(
                    input_channels,
                    out_channels=l['filters'],
                    kernel_size=l['size'],
                    stride=l['stride'],
                    padding=padding,
                    groups=l['groups'],
                    bias=bias,
                ))
                input_channels = l['filters']
                stride *= l['stride']
                if not bias:
                    blocks.add_module('bn', nn.BatchNorm2d(l['filters']))
                if l['activation'] != 'linear':
                    act = ACTIVATION_MAP[l['activation']]()
                    if quant:
                        act = nn.ReLU()
                    blocks.add_module('act', act)
            elif name == 'shortcut':
                blocks = ShortCut(l['activation'], quant)
                setattr(blocks, '_from', l['from'])
                setattr(layers[-1], '_notprune', True)
                setattr(layers[l['from']], '_notprune', True)
            elif name == 'route':
                blocks = Route(quant)
                layer_indexes = l['layers']
                if isinstance(layer_indexes, int):
                    layer_indexes = [layer_indexes]
                setattr(blocks, '_layers', layer_indexes)
                input_channels = sum(layers[li]._output_channels for li in layer_indexes)
                stride = layers[layer_indexes[0]]._stride
                strides = [not(layers[li]._stride - stride) for li in layer_indexes]
                assert all(strides), 'not all route layer strides is same'
            elif name == 'maxpool':
                padding = l['size'] // 2 if l['pad'] == 1 else l['padding']
                blocks = nn.MaxPool2d(l['size'], stride=l['stride'], padding=padding)
                stride *= l['stride']
            elif name == 'upsample':
                blocks = nn.UpsamplingNearest2d(scale_factor=l['stride'])
                stride //= l['stride']
            elif name == 'yolo':
                assert l['bbox_loss'] in {'giou', 'iou', 'l1'}, 'unspport bbox loss type in yolo layer'
                opt = l.copy()
                opt['stride'] = stride
                blocks = YOLOLayer(opt)
                setattr(layers[-1], '_notprune', True)

            setattr(blocks, '_output_channels', input_channels)
            setattr(blocks, '_stride', stride)
            setattr(blocks, '_type', name)
            setattr(blocks, '_raw', l)
            layers.append(blocks)
        return layers


if __name__ == "__main__":
    with open('model/cfg/yolov3.cfg', 'r') as fr:
        print(Parser(fr).torch_layers())
