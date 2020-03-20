from collections.abc import Sequence

import torch
from torch import nn


class Baselayer:
    def __init__(self, layer_id: int, input_layers: list, modules: nn.Module, keep_out: bool=False):
        self.layer_id = layer_id
        self.layer_name = modules._type
        self.input_layers = input_layers
        self.modules = modules
        self.out_mask = None
        self.keep_out = keep_out

    def prune(self, threshold) -> int:
        self.out_mask = self.input_layers[0].out_mask
        return 0

    def reflect(self) -> str:
        return self._construct_segment(self.modules._raw)

    @staticmethod
    def _cloneBN(bn, state_dict, mask):
        assert isinstance(bn, nn.BatchNorm2d)
        bn.weight.data = state_dict[0][mask].clone()
        bn.bias.data = state_dict[1][mask].clone()
        bn.running_mean = state_dict[2][mask].clone()
        bn.running_var = state_dict[3][mask].clone()

    @staticmethod
    def _construct_segment(d: dict) -> str:

        def to_str(v) -> str:
            if isinstance(v, str):
                return v
            elif isinstance(v, int) or isinstance(v, float):
                return str(v)
            elif isinstance(v, Sequence):
                return ', '.join(to_str(i) for i in v)
            else:
                raise ValueError("cant parse '{}'(type: {}) back to str".format(v, type(v)))

        head_format = '[{}]'
        head = None
        attr_format = '{}={}'
        body = []
        for key, val in d.items():
            if key == 'name':
                head = head_format.format(val)
            else:
                body.append(attr_format.format(key, to_str(val)))
        assert head is not None, 'cant parse a segment without head'
        return '\n'.join([head]+body)

class Conv2d(Baselayer):
    # 'conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var'
    # or
    # 'conv.weight', 'conv.bias'
    def __init__(self, layer_id: int, input_layers: list, modules: nn.Module, keep_out: bool=False):
        super().__init__(layer_id, input_layers, modules, keep_out)
        self.bn_scale = None
        try:
            bn_layer = self.modules.bn
        except AttributeError:
            pass
        else:
            self.bn_scale = bn_layer.state_dict()['weight'].abs().clone()

    def prune(self, threshold) -> int:
        state_dict = list(self.modules.state_dict().values())
        device = state_dict[0].device

        if len(self.input_layers) == 0:
            input_mask = torch.ones(state_dict[0].shape[1]).bool().to(device)
        elif len(self.input_layers) == 1:
            input_mask = self.input_layers[0].out_mask
        else:
            raise ValueError('input of conv layer must be 0 or 1')
        conv_layer = self.modules.conv

        # conv with bias
        if self.bn_scale is None:
            conv_layer.weight.data = state_dict[0][:, input_mask, :, :].clone()
            conv_layer.bias.data = state_dict[1].clone()
            self.out_mask = torch.ones(state_dict[0].shape[0]).bool().to(device)
            return 0

        # deepwise conv+bn
        bn_layer = self.modules.bn
        if conv_layer.groups > 1:
            assert conv_layer.groups == conv_layer.in_channels, 'only support deepwise conv'
            conv_layer.weight.data = state_dict[0][input_mask, :, :, :].clone()
            conv_layer.groups = input_mask.sum().item()
            self._cloneBN(bn_layer, state_dict[1:5], input_mask)
            self.out_mask = input_mask
            return 0

        temp = state_dict[0][:, input_mask, :, :]
        # keepout conv+bn
        if self.keep_out:
            conv_layer.weight.data = temp.clone()
            self._cloneBN(bn_layer, state_dict[1:5], slice(None))
            self.out_mask = torch.ones(state_dict[1].shape[0]).bool().to(device)
            return 0
        # normal conv+bn
        divisor = 8
        thres_index = ((self.bn_scale.gt(threshold).sum().item() + divisor - 1) // divisor) * divisor
        picked_bn_indexes = torch.sort(self.bn_scale, descending=True)[1][:thres_index]
        prune_mask = torch.zeros_like(self.bn_scale, dtype=torch.bool)
        prune_mask[picked_bn_indexes] = 1
        conv_layer.weight.data = temp[prune_mask, :, :, :].clone()
        self._cloneBN(bn_layer, state_dict[1:5], prune_mask)
        self.out_mask = prune_mask
        return len(self.bn_scale) - thres_index

    def reflect(self) -> str:
        d = self.modules._raw
        conv_layer = self.modules.conv
        d['filters'] = conv_layer.state_dict()['weight'].shape[0]
        d['groups'] = conv_layer.groups
        return self._construct_segment(d)

class ShortCut(Baselayer):

    def prune(self, threshold) -> int:
        masks = [l.out_mask for l in self.input_layers]
        fit = all(torch.eq(x, y).all() for x, y in zip(masks, masks[1:]))
        assert fit, 'not all layers outmask is same'
        self.out_mask = masks[0]
        return 0

class Route(Baselayer):

    def prune(self, threshold) -> int:
        self.out_mask = torch.cat([l.out_mask for l in self.input_layers])
        return 0

class Pool(Baselayer):
    pass

class Upsample(Baselayer):
    pass

class YOLO(Baselayer):
    pass
