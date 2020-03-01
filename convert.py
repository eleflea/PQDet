import numpy as np
import torch

def save_weights(weight_path: str, save_path: str, seen: int=0):
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

if __name__ == "__main__":
    save_weights('weights/model-74-0.7724.pt', 'weights/model-74-0.7724_convert.weight')
