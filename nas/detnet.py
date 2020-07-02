import numpy as np
import torch
from torch import nn

import model
from model.interpreter import _TARGET_MAP
from nas.reglayers import ResBottleneckBlock
from nas.regnet import RegNet, adjust_ws_gs_comp, regnet_600M_config

_YOLO_OPT = {
    'classes': 20,
    'bbox_loss': 'giou',
    'ignore_thresh': 0.5,
    'l1_loss_gain': 0.05,
}

def quantize_float(f: np.ndarray, q):
    """Converts a float to closest non-zero int divisible by q."""
    return (np.round(f / q) * q).astype(np.int)

def _get_stage_outchannels(b: nn.Module):
    return list(b.children())[-1].f.c.out_channels

def generate_cfg(merges):
    # w_outs = quantize_float(np.exp2(np.random.uniform(low=5., high=10., size=(3, ))), 8)
    # ds = [np.round(np.exp2(np.random.uniform(low=0., high=3.))).astype(np.int)] * 3
    # ps = [np.random.randint(low=1, high=ds[0] + 1)] * 3
    # bms = [np.random.choice([1/4, 1/2, 1, 2, 4])] * 3
    # gws = [np.random.choice([1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48])] * 3
    # w_outs, gws = adjust_ws_gs_comp(w_outs, bms, gws)
    # w_ins = [merges[0], w_outs[0], w_outs[1]]
    # strides = [32, 16, 8]
    # merges[0] = 0
    u = np.random.uniform
    rs = np.log2(np.array([(96, 1024), (120, 1024), (32, 768)]))
    w_outs = quantize_float(np.exp2(np.array([u(low=r[0], high=r[1]) for r in rs])), 8)
    ds = [np.round(np.exp2(np.random.uniform(low=0., high=2.585))).astype(np.int)] * 3
    ps = [np.random.randint(low=1, high=ds[0] + 1)] * 3
    bms = [np.random.choice([1/2, 1, 2])] * 3
    gws = [np.random.choice([1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48])] * 3
    w_outs, gws = adjust_ws_gs_comp(w_outs, bms, gws)
    w_ins = [merges[0], w_outs[0], w_outs[1]]
    strides = [32, 16, 8]
    merges[0] = 0

    return [dict(zip(('w_in', 'w_out', 'd', 'p', 'bm', 'gw', 'stride', 'merge'), params))
        for params in zip(w_ins, w_outs, ds, ps, bms, gws, strides, merges)]

class DetBranch(nn.Module):

    def __init__(self, w_in, w_out, d, p, bm, gw, stride, merge=0):
        super(DetBranch, self).__init__()
        assert p > 0 and p <= d, 'branch position must in depth'
        self.p = p
        if merge:
            self.add_module('merge', Merge(w_in, w_out))
            w_in = w_out + merge
            self.p += 1
        for i in range(d):
            self.add_module(
                f'b{i + 1}',
                ResBottleneckBlock(w_in, w_out, 1, bm, gw)
            )
            w_in = w_out
        self.add_module('head', nn.Conv2d(w_in, 75, 1, 1, padding=0, bias=True))
        opt = _YOLO_OPT.copy()
        opt['stride'] = stride
        self.add_module('yolo', model.parser.YOLOLayer(opt))

    def forward(self, x, target=None):
        blocks = list(self.children())
        merge_out = None
        for i, layer in enumerate(blocks[:-1]):
            if i == self.p:
                merge_out = x
            x = layer(x)
        assert merge_out is not None
        out = self.yolo(x, target)
        return out, merge_out

class Merge(nn.Module):

    def __init__(self, w_in, w_out):
        super(Merge, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(w_in, w_out, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(w_out),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, xs):
        x, y = xs
        x = self.upsample(self.conv(x))
        return torch.cat([x, y], dim=1)

class DetHead(nn.Module):

    def __init__(self, cfgs):
        super(DetHead, self).__init__()
        self.y1 = DetBranch(**cfgs[0])
        self.y2 = DetBranch(**cfgs[1])
        self.y3 = DetBranch(**cfgs[2])

    def forward(self, xs, target=None):
        o1, y1 = self.y1(xs[2], target=_TARGET_MAP[32](target))
        o2, y2 = self.y2([y1, xs[1]], target=_TARGET_MAP[16](target))
        o3, _ = self.y3([y2, xs[0]], target=_TARGET_MAP[8](target))
        outputs = [o1, o2, o3]

        if target is None:
            outputs = [o.view((o.shape[0], -1, o.shape[-1])) for o in outputs]
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

class DetNet(RegNet):

    def __init__(self, regnet_cfg, fpn_cfg=None, regnet_weight_path='', **kwargs):
        super(DetNet, self).__init__(regnet_cfg, **kwargs)

        if regnet_weight_path:
            sd = torch.load(regnet_weight_path)
            nn.DataParallel(self).load_state_dict(sd)

        if fpn_cfg is None:
            merges = [_get_stage_outchannels(getattr(self, f's{i}')) for i in range(4, 1, -1)]
            fpn_cfg = generate_cfg(merges)
        self.head = DetHead(fpn_cfg)
        for m in self.head.modules():
            if isinstance(m, nn.BatchNorm2d) and hasattr(m, "final_bn") and m.final_bn:
                m.weight.data.zero_()
        self.cfg = fpn_cfg

    def forward(self, x, target=None):
        b_outs = []
        x = self.stem(x)
        x = self.s1(x)
        for i in range(2, 5):
            x = getattr(self, f's{i}')(x)
            b_outs.append(x)
        return self.head(b_outs, target)

def detnet_600m(fpn_cfg=None, pretrained=True):
    if pretrained:
        return DetNet(
            regnet_600M_config, fpn_cfg,
            regnet_weight_path='weights/pretrained/regnet_600m_741.pth'
        )
    return DetNet(regnet_600M_config, fpn_cfg)

if __name__ == "__main__":
    from thop import clever_format, profile
    import tools
    net = detnet_600m().cuda()
    inputs = torch.randn(1, 3, 512, 512).cuda()
    flops, params = profile(net, inputs=(inputs, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print('MACs: {}, params: {}'.format(flops, params))
    avg_time = tools.compute_time(net, batch_size=16)
    print(f'{avg_time:.2f} ms')
