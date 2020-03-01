from pruning.pruner import BasePruner
from pruning.block import CB, Conv
import torch

class SlimmingPruner(BasePruner):
    def __init__(self, model_cls):
        super().__init__(model_cls)

    def prune(self):
        super().prune()
        # gather BN weights
        bns = []
        maxbn = []
        for b in self.blocks.values():
            if b.bn_scale is not None:
                bns.extend(b.bn_scale.tolist())
                maxbn.append(b.bn_scale.max().item())

        bns = torch.Tensor(bns)
        sorted_bns, i = torch.sort(bns)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.scatter(np.arange(y.shape[0])/y.shape[0],y.numpy())
        # plt.show()
        # assert 0
        prune_limit = (sorted_bns == min(maxbn)).nonzero().item() / len(bns)
        print("prune limit: {}".format(prune_limit))
        if self.prune_ratio > prune_limit:
            raise AssertionError('prune ratio bigger than limit')
        thre_index = int(bns.shape[0] * self.prune_ratio)
        thre = sorted_bns[thre_index]
        thre = thre.cuda()
        pruned_bn = 0
        for b in self.blocks.values():
            if not(isinstance(b, CB) or isinstance(b, Conv)):
                continue
            if b.bn_scale is not None:
                mask = b.bn_scale.gt(thre)
                pruned_bn = pruned_bn + mask.shape[0] - torch.sum(mask).item()
                b.prune_mask = torch.where(mask == 1)[0]
                # b.prune_mask = torch.arange(b.bn_scale.shape[0])
            if len(b.input_layer) == 1:
                input_mask = torch.arange(b.input_channel) if b.input_layer[0] is None else b.input_layer[0].out_mask
            elif len(b.input_layer) == 2:
                first = b.input_layer[0].out_mask
                second = b.input_layer[1].out_mask
                second += b.input_layer[0].output_channel
                second = second.to(first.device)
                input_mask = torch.cat((first, second), 0)
            else:
                raise AttributeError
            b.clone2module(input_mask)
            print("{}: {}/{} pruned".format(b.layer_name, mask.shape[0] - torch.sum(mask).item(), mask.shape[0]))
        print("Slimming Pruner done")
