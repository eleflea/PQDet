from collections import namedtuple
from typing import Sequence, List, Optional

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from torch import nn
from model.parser import build_center_grid

_IDENTITY_LAYER = nn.Identity()
_QUANTIZED_BINARY_OP_PARAM_MAP = {
    'add': lambda x: x._from,
    'cat': lambda x: x._layers[1],
}

def _check_conv_layer_identity(layer: nn.Module):
    assert type(getattr(layer, 'bn', _IDENTITY_LAYER)) == nn.Identity,\
        'bn in quantized model must be identity for onnx export'
    assert type(getattr(layer, 'act', _IDENTITY_LAYER)) == nn.Identity,\
        'act in quantized model must be identity for onnx export'

def _make_initializers(names: Sequence[str], values: Sequence[np.ndarray]):
    initializers = []
    for name, value in zip(names, values):
        init = numpy_helper.from_array(value, name)
        initializers.append(init)
    return initializers

QParameter = namedtuple('QParameter', ['scale', 'zero_point'])
TensorInfo = namedtuple('TensorInfo', ['name', 'qparam'])

class ONNXExporter:

    def __init__(self, model: nn.Module, input_name: str='input',
        output_name: str='output', size=(512, 512)):
        self.model = model
        self.size = np.array(size, dtype=np.int)
        self.input_name = input_name
        self.output_name = output_name

        self.current_output_name = input_name
        self.current_qparam_name = None
        self.layer_output_info = []
        self.nodes = []
        self.initializer = []
        self.yolo_names = []

        self.node_map = {
            'convolutional': self.make_node_qconv,
            'shortcut': self.make_node_qadd,
            'route': self.make_node_qcat,
            'upsample': self.make_node_resize,
            'yolo': self.make_node_yolo,
        }

    def _inverse_search_initialized_value(self, name: str, dtype=np.float32):
        for init in reversed(self.initializer):
            if init.name == name:
                return np.frombuffer(init.raw_data, dtype=dtype)[0]
        raise RuntimeError(f'cant find "{name}" in initializer')

    def _check_qparam(self):
        assert self.current_qparam_name is not None,\
            'need quantize op before quantized op'

    def _get_qparam(self):
        self._check_qparam()
        return (self.current_qparam_name.scale,
            self.current_qparam_name.zero_point)

    def _set_qparam(self, scale, zero_point):
        self.current_qparam_name = QParameter(scale, zero_point)

    @staticmethod
    def _add_prefix(names: Sequence[str], prefix: str) -> List[str]:
        return [f'{prefix}.{name}' for name in names]

    def make_node_qconv(self, layer: nn.Module, prefix: str=''):
        _check_conv_layer_identity(layer)

        inputs = ['INPUT', 'SCALE', 'ZERO_POINT', 'w', 'w.scale',
            'w.zero_point', 'y.scale', 'y.zero_point', 'b']
        inputs = self._add_prefix(inputs, prefix)
        inputs[0] = self.current_output_name
        inputs[1:3] = self._get_qparam()
        output_name = f'{prefix}.output'
        conv = layer.conv
        node = helper.make_node(
            'QLinearConv',
            inputs=inputs,
            outputs=[output_name],
            pads=conv.padding * 2,
            strides=conv.stride,
            group=conv.groups,
        )
        self.nodes.append(node)

        init_names = inputs[3:]
        sd = conv.state_dict()
        weight = sd['weight']
        weight_scale = weight.q_scale()
        input_scale = self._inverse_search_initialized_value(self.current_qparam_name.scale)
        bias = torch.quantize_per_tensor(
            sd['bias'],
            scale=input_scale * weight_scale,
            zero_point=0,
            dtype=torch.qint32,
        ).int_repr().numpy().astype(np.int32)
        init_values = [
            (weight.int_repr().numpy() + 128).astype(np.uint8),
            np.float32(weight_scale),
            np.uint8(weight.q_zero_point() + 128),
            np.float32(sd['scale'].item()),
            np.uint8(sd['zero_point'].item()),
            bias,
        ]
        self.initializer.extend(_make_initializers(init_names, init_values))

        self._set_qparam(*init_names[-3:-1])
        self.layer_output_info.append(TensorInfo(output_name, self.current_qparam_name))
        self.current_output_name = output_name

    def make_node_quant(self, layer: nn.Module, prefix: str=''):
        inputs = ['INPUT', 'scale', 'zero_point']
        inputs = self._add_prefix(inputs, prefix)
        inputs[0] = self.current_output_name
        output_name = f'{prefix}.output'
        node = helper.make_node(
            'QuantizeLinear',
            inputs=inputs,
            outputs=[output_name],
        )
        self.nodes.append(node)

        init_names = inputs[1:]
        sd = layer.state_dict()
        init_values = [
            np.float32(sd['scale'][0].item()),
            np.uint8(sd['zero_point'][0].item()),
        ]
        self.initializer.extend(_make_initializers(init_names, init_values))

        self._set_qparam(*init_names)
        self.current_output_name = output_name

    def make_node_dequant(self, layer: Optional[nn.Module], prefix: str=''):
        inputs = ['INPUT', 'SCALE', 'ZERO_POINT']
        inputs = self._add_prefix(inputs, prefix)
        inputs[0] = self.current_output_name
        inputs[1:3] = self._get_qparam()
        output_name = f'{prefix}.output'
        node = helper.make_node(
            'DequantizeLinear',
            inputs=inputs,
            outputs=[output_name],
        )
        self.nodes.append(node)

        self.current_qparam_name = None
        self.current_output_name = output_name

    def _make_node_qbin_op(self, layer: nn.Module, mode: str, prefix: str=''):
        assert mode in _QUANTIZED_BINARY_OP_PARAM_MAP.keys(),\
            f'"{mode}" is unsupport quantized binary operator'
        dequant_inputs1 = [self.current_output_name, *self._get_qparam()]
        other_tensor_index = _QUANTIZED_BINARY_OP_PARAM_MAP[mode](layer)
        other_tensor_info = self.layer_output_info[other_tensor_index]
        dequant_inputs2 = [
            other_tensor_info.name,
            other_tensor_info.qparam.scale,
            other_tensor_info.qparam.zero_point
        ]
        inner_outputs = self._add_prefix(['{}.output'.format(i) for i in range(3)], prefix)
        quant_inputs = [inner_outputs[-1], *self._add_prefix(['scale', 'zero_point'], prefix)]
        output_name = f'{prefix}.output'
        dequant_node1 = helper.make_node(
            'DequantizeLinear',
            inputs=dequant_inputs1,
            outputs=inner_outputs[:1],
        )
        dequant_node2 = helper.make_node(
            'DequantizeLinear',
            inputs=dequant_inputs2,
            outputs=inner_outputs[1:2],
        )
        if mode == 'add':
            binary_node = helper.make_node(
                'Add',
                inputs=inner_outputs[:2],
                outputs=inner_outputs[2:],
            )
        elif mode == 'cat':
            binary_node = helper.make_node(
                'Concat',
                inputs=inner_outputs[:2],
                outputs=inner_outputs[2:],
                axis=1,
            )
        quant_node = helper.make_node(
            'QuantizeLinear',
            inputs=quant_inputs,
            outputs=[output_name],
        )
        self.nodes.extend([dequant_node1, dequant_node2, binary_node, quant_node])

        init_names = quant_inputs[1:]
        qbin = layer.ffunc
        init_values = [
            np.float32(qbin.scale),
            np.uint8(qbin.zero_point),
        ]
        self.initializer.extend(_make_initializers(init_names, init_values))

        self._set_qparam(*init_names)
        self.layer_output_info.append(TensorInfo(output_name, self.current_qparam_name))
        self.current_output_name = output_name

    def make_node_qadd(self, layer: nn.Module, prefix: str=''):
        self._make_node_qbin_op(layer, 'add', prefix)

    def make_node_qcat(self, layer: nn.Module, prefix: str=''):
        layer_indexes = layer._layers
        if len(layer_indexes) == 1:
            tensor_info = self.layer_output_info[layer_indexes[0]]
            self.current_qparam_name = tensor_info.qparam
            self.layer_output_info.append(tensor_info)
            self.current_output_name = tensor_info.name
            return
        assert len(layer_indexes) == 2, 'only support 2 element concat now'
        self._make_node_qbin_op(layer, 'cat', prefix)

    def make_node_resize(self, layer: nn.Module, prefix: str=''):
        inputs = ['INPUT', 'empty', 'scales']
        inputs = self._add_prefix(inputs, prefix)
        inputs[0] = self.current_output_name
        output_name = f'{prefix}.output'
        node = helper.make_node(
            'Resize',
            inputs=inputs,
            outputs=[output_name],
            mode='nearest',
        )
        self.nodes.append(node)

        init_names = inputs[1:]
        sf = layer.scale_factor
        init_values = [
            np.array([], dtype=np.float32),
            np.array([1, 1, sf, sf], dtype=np.float32),
        ]
        self.initializer.extend(_make_initializers(init_names, init_values))

        self.layer_output_info.append(TensorInfo(output_name, self.current_qparam_name))
        self.current_output_name = output_name

    def make_node_yolo(self, layer: nn.Module, prefix: str=''):
        self.make_node_dequant(None, f'{prefix}.dequant')
        inner_outputs = self._add_prefix([
            'transpose', 'reshape.0', 'spilt.0', 'spilt.1', 'exp',
            'split.0.0', 'spilt.0.1', 'sub', 'add', 'cat.0', 'mul',
            'sigmoid', 'cat',
        ], f'{prefix}.decode')
        init_names = self._add_prefix(
            ['reshape.0.shape', 'xy_grid', 'stride', 'reshape.1.shape'],
            f'{prefix}.decode',
        )
        output_name = f'{prefix}.output'

        num_classes = layer._raw['classes']
        gt_per_scale = layer._output_channels // (num_classes + 5)
        grid_size = self.size // layer._stride

        transpose_node = helper.make_node(
            'Transpose',
            inputs=[self.current_output_name],
            outputs=inner_outputs[:1],
            perm=[0, 2, 3, 1],
        )
        reshape_node1 = helper.make_node(
            'Reshape',
            inputs=[inner_outputs[0], init_names[0]],
            outputs=inner_outputs[1:2],
        )
        split_node1 = helper.make_node(
            'Split',
            inputs=inner_outputs[1:2],
            outputs=inner_outputs[2:4],
            axis=-1,
            split=[4, num_classes + 1],
        )
        exp_node = helper.make_node(
            'Exp',
            inputs=inner_outputs[2:3],
            outputs=inner_outputs[4:5],
        )
        split_node2 = helper.make_node(
            'Split',
            inputs=inner_outputs[4:5],
            outputs=inner_outputs[5:7],
            axis=-1,
        )
        sub_node = helper.make_node(
            'Sub',
            inputs=[init_names[1], inner_outputs[5]],
            outputs=inner_outputs[7:8],
        )
        add_node = helper.make_node(
            'Add',
            inputs=[init_names[1], inner_outputs[6]],
            outputs=inner_outputs[8:9],
        )
        cat_node1 = helper.make_node(
            'Concat',
            inputs=inner_outputs[7:9],
            outputs=inner_outputs[9:10],
            axis=-1,
        )
        mul_node = helper.make_node(
            'Mul',
            inputs=[inner_outputs[9], init_names[2]],
            outputs=inner_outputs[10:11],
        )
        sigmoid_node = helper.make_node(
            'Sigmoid',
            inputs=inner_outputs[3:4],
            outputs=inner_outputs[11:12],
        )
        cat_node2 = helper.make_node(
            'Concat',
            inputs=inner_outputs[10:12],
            outputs=inner_outputs[12:],
            axis=-1,
        )
        reshape_node2 = helper.make_node(
            'Reshape',
            inputs=[inner_outputs[12], init_names[3]],
            outputs=[output_name],
        )
        self.nodes.extend([
            transpose_node, reshape_node1, split_node1, exp_node, split_node2,
            sub_node, add_node, cat_node1, mul_node, sigmoid_node, cat_node2, reshape_node2,
        ])

        init_values = [
            np.array([0, 0, 0, gt_per_scale, -1], dtype=np.int64),
            build_center_grid(*grid_size).numpy().astype(np.float32),
            np.float32(layer._stride),
            np.array([0, -1, num_classes + 5], dtype=np.int64),
        ]
        self.initializer.extend(_make_initializers(init_names, init_values))

        self.layer_output_info.append(TensorInfo(output_name, self.current_qparam_name))
        self.current_output_name = output_name
        self.yolo_names.append(output_name)

    def make_node_output(self):
        node = helper.make_node(
            'Concat',
            inputs=self.yolo_names,
            outputs=[self.output_name],
            axis=1,
        )
        self.nodes.append(node)
        self.current_output_name = self.output_name

    def _build_graph(self, graph_name: str):
        # pylint: disable-msg=no-member
        graph_input = helper.make_tensor_value_info(
            self.input_name, TensorProto.FLOAT, [-1, 3] + self.size.tolist()
        )
        graph_output = helper.make_tensor_value_info(
            self.output_name, TensorProto.FLOAT, [-1, 16128, 25]
        )
        graph_output1 = helper.make_tensor_value_info(
            '35.output', TensorProto.UINT8, [-1, -1, -1, -1]
        )
        # pylint: enable-msg=no-member
        graph = helper.make_graph(
            nodes=self.nodes,
            name=graph_name,
            inputs=[graph_input],
            outputs=[graph_output, graph_output1],
            initializer=self.initializer,
        )
        onnx_model = helper.make_model(graph, producer_name='eleflea')
        onnx.checker.check_model(onnx_model)
        return onnx_model

    def export(self, graph_name: str='mobilenet-yolo'):
        self.make_node_quant(self.model.qstub, 'quant')
        for i, layer in enumerate(self.model.module_list):
            node_func = self.node_map.get(layer._type)
            if node_func is None:
                raise NotImplementedError(f'{layer._type} onnx export not support')
            node_func(layer, str(i))
        self.make_node_output()
        return self._build_graph(graph_name)
