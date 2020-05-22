import numpy as np
import onnx
import torch

import tools
from export.onnx_exporter import ONNXExporter


def save_weight_to_darknet(weight_path: str, save_path: str, seen: int=0):
    fw = open(save_path, 'wb')

    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))['model']

    header = np.array([0, 0, 0, seen], dtype=np.int32)
    header.tofile(fw)

    pre_conv = None
    pre_bias = []
    for key, params in state_dict.items():
        params_shape = len(params.shape)
        if params_shape == 4: # conv weight
            if pre_conv is not None:
                pre_conv.numpy().tofile(fw)
            pre_conv = params
        elif params_shape == 1: #BN or conv_bias
            if key.endswith('bias') and len(pre_bias) == 0: # conv_bias
                params.numpy().tofile(fw)
                pre_conv.numpy().tofile(fw)
                pre_conv = None
            else: # BN
                pre_bias.append(params)
                if len(pre_bias) == 4:
                    pre_bias[1].numpy().tofile(fw)
                    pre_bias[0].numpy().tofile(fw)
                    pre_bias[2].numpy().tofile(fw)
                    pre_bias[3].numpy().tofile(fw)
                    pre_bias.clear()
                    assert pre_conv is not None
                    pre_conv.numpy().tofile(fw)
                    pre_conv = None
        else:
            pass

    if pre_conv is not None:
        pre_conv.numpy().tofile(fw)

    fw.close()

def export_quantized_to_onnx(cfg_path: str, weight_path: str, onnx_path: str):
    model = tools.build_model(
        cfg_path, weight_path, device='cpu', dataparallel=False, quantized=True, backend='qnnpack'
    )[0]
    model.eval()
    onnx_exporter = ONNXExporter(model, size=(512, 512))
    onnx_model = onnx_exporter.export(graph_name='quantized-mobilenetv2-yolov3-lite')
    onnx.save(onnx_model, onnx_path)

def export_normal_to_onnx(cfg_path: str, weight_path: str, onnx_path: str):
    model = tools.build_model(
        cfg_path, weight_path, device='cpu', dataparallel=False
    )[0]
    model.eval()

    torch_in = torch.randn(1, 3, 512, 512)
    dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    torch.onnx.export(
        model, torch_in, onnx_path, verbose=False, input_names=['input'],
        output_names=['output'], dynamic_axes=dynamic_axes, opset_version=11,
    )

def partial(weight_path: str, save_path: str, layers: int):
    state_dict = torch.load(weight_path, 'cpu')['model']
    partial_dict = {}
    sentinel = f'{layers+1}.'
    for key, params in state_dict.items():
        if sentinel in key:
            break
        partial_dict[key] = params
    torch.save(partial_dict, save_path)

def make_backbone(weight_path: str, cfg_path: str, save_path: str):
    state_dict = torch.load(weight_path, 'cpu')['state_dict']
    model = tools.build_model(cfg_path, device='cpu', dataparallel=False)[0]
    new_state_dict = {}
    for (bn, bp), (mn, mp) in zip(state_dict.items(), model.state_dict().items()):
        if not bp.shape == mp.shape:
            print(f'last layer: {bn}({list(bp.shape)}) -> {mn}({list(mp.shape)})')
            break
        new_state_dict[mn] = bp
    torch.save(new_state_dict, save_path)

if __name__ == "__main__":
    weight_path = 'weights/VOC_std_prune40_quant/model-77-0.7674.pt'
    # partial(weight_path, 'weights/pretrained/mobilev2-prune40.pt', 61)
    # weight_path = 'weights/VOC_quant3/model-44.pt'
    # weight_path = 'weights/trained/model-74-0.7724.pt'
    # save_weight_to_darknet(weight_path, weight_path.rsplit('.', 1)[0]+'-convert.weights')
    # export_quantized_to_onnx('model/cfg/myolo-prune-40.cfg', weight_path, 'export/quant_myolov1.onnx')
    # export_normal_to_onnx('model/cfg/myolo-prune-40.cfg', weight_path, 'export/myolo-prune40.onnx')
    make_backbone('/home/eleflea/code/classifier/regnet_600m_741.pth', 'model/cfg/regnetx-600m-yolo.cfg', 'weights/pretrained/regnetx_600m.pt')
