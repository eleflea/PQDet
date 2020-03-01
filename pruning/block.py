import torch
from torch import nn

class Baselayer:
    def __init__(self, layer_name: str, layer_id: int, inputs: list, modules: list, state_dict: list, keep_output: bool=False):
        self.layer_name = layer_name
        self.layer_id = layer_id
        self.input_layer = inputs
        self.modules = modules
        self.inputchannel = 0
        self.outputchannel = 0
        # filter relu
        self.state_dict = [s for s in state_dict if len(s.shape) != 0]
        self.prune_mask = None
        self.out_mask = None
        self.bn_scale = None
        self.keep_output = keep_output
    def clone2module(self, input_mask):
        raise NotImplementedError

    def _cloneBN(self, bn, state_dict, mask):
        assert isinstance(bn, nn.BatchNorm2d)
        bn.weight.data = state_dict[0][mask].clone()
        bn.bias.data = state_dict[1][mask].clone()
        bn.running_mean = state_dict[2][mask].clone()
        bn.running_var = state_dict[3][mask].clone()

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "name={}, ".format(self.layer_name)
        s += "id={}, ".format(self.layer_id)
        s += "numweights={},".format(len(self.state_dict))
        s += "inchannel={},".format(self.input_channel)
        s += "outchannel={})".format(self.output_channel)
        return s

class CB(Baselayer):
    def __init__(self, layer_name: str, layer_id: int, inputs: list, modules: list, state_dict: list, keep_output: bool=False):
        super().__init__(layer_name, layer_id, inputs, modules, state_dict, keep_output)
        # 'conv.weight', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var'
        self.input_channel = self.state_dict[0].shape[1]
        self.output_channel = self.state_dict[-1].shape[0]
        self.bn_scale = self.state_dict[1].abs().clone()

    def clone2module(self, input_mask):
        conv_layer, bn_layer = self.modules
        if conv_layer.groups > 1:
            conv_layer.weight.data = self.state_dict[0][input_mask, :, :, :].clone()
            conv_layer.groups = input_mask.shape[0]
            self._cloneBN(bn_layer, self.state_dict[1:5], input_mask)
            self.out_mask = input_mask
            return

        temp = self.state_dict[0][:, input_mask, :, :]
        if self.keep_output:
            conv_layer.weight.data = temp.clone()
            self._cloneBN(bn_layer, self.state_dict[1:5], None)
            self.out_mask = torch.arange(self.state_dict[1].shape[0])
        else:
            conv_layer.weight.data = temp[self.prune_mask, :, :, :].clone()
            self._cloneBN(bn_layer, self.state_dict[1:5], self.prune_mask)
            self.out_mask = self.prune_mask

class Conv(Baselayer):
    def __init__(self, layer_name: str, layer_id: int, inputs: list, modules: list, state_dict: list, keep_output: bool=False):
        super().__init__(layer_name, layer_id, inputs, modules, state_dict)
        # 'conv.weight'
        self.input_channel = self.state_dict[0].shape[1]
        self.output_channel = self.state_dict[0].shape[0]

    def clone2module(self, input_mask):
        conv_layer = self.modules[0]
        conv_layer.weight.data = self.state_dict[0][:, input_mask.tolist(), :, :].clone()
        conv_layer.bias.data = self.state_dict[1].clone()
        self.out_mask = torch.arange(self.state_dict[0].shape[0])
