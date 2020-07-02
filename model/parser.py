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

class LayerInfo:

    def __init__(self):
        # out_channels, height, width
        self.shape = [None, None, None]
        self.stride = 1

    def update_stride(self, stride, down=False):
        if self.stride is None:
            return
        if down:
            self.stride //= stride
        else:
            self.stride *= stride

    def set_size(self, size):
        self.shape[1:] = size

    @property
    def channels(self):
        return self.shape[0]

    @channels.setter
    def channels(self, channels):
        self.shape[0] = channels

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
    'fc': {
        'name': 'fc',
        'input': 1,
        'output': 1,
        'activation': 'logistic',
    },
    'shortcut': {
        'name': 'shortcut',
        'activation': 'linear',
        'alpha': 1, # not use
        'beta': 1, # not use
    },
    'scale_channels': {
        'name': 'scale_channels',
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
    'avgpool': {
        'name': 'avgpool',
        'height': 1,
        'width': 1,
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
    },
    'dropout': {
        'probability': 0.5,
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

class FC(nn.Module):
    def __init__(self, input: int, output: int, activation: str):
        super().__init__()
        self.fc = nn.Linear(input, output)
        self.act = None
        if activation != 'linear':
            self.act = ACTIVATION_MAP[activation]()

        self.input = input

    def forward(self, x):
        x = x.view(self.input)
        x = self.fc(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ShortCut(nn.Module):
    def __init__(self, activation: str, quant: bool=False):
        super().__init__()
        self.quant = quant
        if quant:
            self.ffunc = FloatFunctional()
        self.act = None
        if activation != 'linear':
            self.act = ACTIVATION_MAP[activation]()

    def forward(self, x, other):
        if self.quant:
            x = self.ffunc.add(x, other)
        else:
            x += other
        if self.act is not None:
            x = self.act(x)
        return x

class ScaleChannels(nn.Module):
    def __init__(self, quant: bool=False):
        super().__init__()
        self.quant = quant
        if quant:
            self.ffunc = FloatFunctional()

    def forward(self, x, other):
        if self.quant:
            return self.ffunc.mul(x, other)
        return other * x

class Route(nn.Module):
    def __init__(self, quant: bool=False, single: bool=False):
        super().__init__()
        self.quant = quant
        self.single = single
        if not single:
            self.ffunc = FloatFunctional()

    def forward(self, xs):
        if self.single:
            return xs[0]
        if self.quant:
            return self.ffunc.cat(xs, dim=1)
        return torch.cat(xs, dim=1)

def build_center_grid(height, width) -> torch.Tensor:
    shiftx = torch.arange(0, height, dtype=torch.float32) + 0.5
    shifty = torch.arange(0, width, dtype=torch.float32) + 0.5
    shifty, shiftx = torch.meshgrid(shiftx, shifty)
    shiftx = shiftx.unsqueeze(-1)
    shifty = shifty.unsqueeze(-1)
    xy_grid = torch.stack([shiftx, shifty], dim=-1)
    return xy_grid

class Decode(nn.Module):
    def __init__(self, num_classes: int, stride: int, onnx: bool=False):
        super(Decode, self).__init__()
        self.num_classes = num_classes
        self.stride = stride
        self.onnx = onnx
        self.grid_size = (0, 0)

    def respawn_grid(self, height, width, device):
        self.xy_grid = build_center_grid(height, width).to(device)
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

        if self.onnx:
            xy_grid = build_center_grid(out_size_h, out_size_w).to(conv.device)
        else:
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

    def __init__(self, opt: dict, onnx: bool=False):
        super().__init__()
        self.decode = Decode(opt['classes'], opt['stride'], onnx)
        self.opt = opt

    def forward(self, x, target=None):
        x = self.decode(x)
        if target is None:
            return x
        losses = loss_per_scale(x, *target, self.opt)
        return losses

def _solve_padding(size: int, padding: int, pad: Union[bool, int]):
    return size // 2 if bool(pad) else padding

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

    def torch_layers(self, quant: bool=False, onnx: bool=False) -> List[nn.Module]:

        def layer_index(where: int=0):
            return len(layers) + where + 1 if where <= 0 else where + 1

        def assert_channels_match(index1: int, index2: int):
            num_chan1 = layers[index1]._output_channels
            num_chan2 = layers[index2]._output_channels
            assert num_chan1 == num_chan2,\
                '{} layer[{}]: out channels dont match between layer {}({}) and {}({})'.format(
                    name, layer_index(), layer_index(index1), num_chan1,
                    layer_index(index2), num_chan2
                )

        def assert_strides_match(indexes):
            s0 = layers[indexes[0]]._stride
            strides = [not(layers[idx]._stride - s0) for idx in indexes]
            assert all(strides), '{} layer[{}]: not all input strides is same: {}'.format(
                name, layer_index(), [layers[idx]._stride for idx in indexes]
            )

        layer_info = LayerInfo()
        layers = []
        for l in self.gen_layers():
            name = l['name']
            if name == 'net':
                layer_info.channels = l['channels']
                continue
            if name == 'convolutional':
                blocks = nn.Sequential()
                padding = _solve_padding(l['size'], l['padding'], l['pad'])
                bias = l['batch_normalize'] == 0
                blocks.add_module('conv', nn.Conv2d(
                    layer_info.channels,
                    out_channels=l['filters'],
                    kernel_size=l['size'],
                    stride=l['stride'],
                    padding=padding,
                    groups=l['groups'],
                    bias=bias,
                ))
                layer_info.channels = l['filters']
                layer_info.update_stride(l['stride'])
                if not bias:
                    blocks.add_module('bn', nn.BatchNorm2d(l['filters']))
                if l['activation'] != 'linear':
                    act = ACTIVATION_MAP[l['activation']]()
                    if quant:
                        act = nn.ReLU()
                    blocks.add_module('act', act)
            elif name == 'fc':
                blocks = FC(l['input'], l['output'], l['activation'])
                layer_info.channels = l['output']
                setattr(layers[-1], '_notprune', True)
            elif name == 'shortcut':
                assert_channels_match(-1, l['from'])
                blocks = ShortCut(l['activation'], quant)
                # init gamma of final bn to zero
                # if l['activation'] != 'linear' and layers[-1]._type == 'convolutional'\
                #     and hasattr(layers[-1], 'bn'):
                #     layers[-1].bn.weight.data.zero_()
                setattr(blocks, '_from', l['from'])
                setattr(layers[-1], '_notprune', True)
                setattr(layers[l['from']], '_notprune', True)
            elif name == 'scale_channels':
                assert_channels_match(-1, l['from'])
                blocks = ScaleChannels(quant)
                setattr(blocks, '_from', l['from'])
                layer_info.stride = layers[l['from']]._stride
            elif name == 'route':
                layer_indexes = l['layers']
                single = False
                if isinstance(layer_indexes, int):
                    single = True
                    layer_indexes = [layer_indexes]
                assert_strides_match(layer_indexes)
                blocks = Route(quant, single)
                setattr(blocks, '_layers', layer_indexes)
                layer_info.channels = sum(layers[li]._output_channels for li in layer_indexes)
                layer_info.stride = layers[layer_indexes[0]]._stride
            elif name == 'maxpool':
                padding = _solve_padding(l['size'], l['padding'], l['pad'])
                blocks = nn.MaxPool2d(l['size'], stride=l['stride'], padding=padding)
                layer_info.update_stride(l['stride'])
            elif name == 'avgpool':
                size = (l['height'], l['width'])
                blocks = nn.AdaptiveAvgPool2d(size)
                layer_info.stride = None
                layer_info.set_size(size)
            elif name == 'upsample':
                blocks = nn.UpsamplingNearest2d(scale_factor=l['stride'])
                layer_info.update_stride(l['stride'], down=True)
            elif name == 'yolo':
                assert l['bbox_loss'] in {'diou', 'ciou', 'giou', 'iou', 'l1'},\
                    'unspport bbox loss type in yolo layer: {}'.format(l['bbox_loss'])
                opt = l.copy()
                opt['stride'] = layer_info.stride
                blocks = YOLOLayer(opt, onnx)
                setattr(layers[-1], '_notprune', True)
            elif name == 'dropout':
                blocks = nn.Dropout(p=l['probability'])
            else:
                raise ValueError(f"unsupport layer type: '{name}'")

            setattr(blocks, '_output_channels', layer_info.channels)
            setattr(blocks, '_stride', layer_info.stride)
            setattr(blocks, '_type', name)
            setattr(blocks, '_raw', l)
            layers.append(blocks)
        return layers


if __name__ == "__main__":
    with open('model/cfg/yolov3.cfg', 'r') as fr:
        print(Parser(fr).torch_layers())
