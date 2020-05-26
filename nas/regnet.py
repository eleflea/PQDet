import numpy as np

from nas.reglayers import AnyNet

regnet_200M_config = {'WA': 36.44, 'W0': 24, 'WM': 2.49, 'DEPTH': 13, 'GROUP_W': 8, 'BOT_MUL': 1, 'SE_R': False}
regnet_400M_config = {'WA': 24.48, 'W0': 24, 'WM': 2.54, 'DEPTH': 22, 'GROUP_W': 16, 'BOT_MUL': 1, 'SE_R': False}
regnet_600M_config = {'WA': 36.97, 'W0': 48, 'WM': 2.24, 'DEPTH': 16, 'GROUP_W': 24, 'BOT_MUL': 1, 'SE_R': False}
regnet_800M_config = {'WA': 35.73, 'W0': 56, 'WM': 2.28, 'DEPTH': 16, 'GROUP_W': 16, 'BOT_MUL': 1, 'SE_R': False}
regnet_1600M_config = {'WA': 34.01, 'W0': 80, 'WM': 2.25, 'DEPTH': 18, 'GROUP_W': 24, 'BOT_MUL': 1, 'SE_R': False}
regnet_3200M_config = {'WA': 26.31, 'W0': 88, 'WM': 2.25, 'DEPTH': 25, 'GROUP_W': 48, 'BOT_MUL': 1, 'SE_R': False}
regnet_4000M_config = {'WA': 38.65, 'W0': 96, 'WM': 2.43, 'DEPTH': 23, 'GROUP_W': 40, 'BOT_MUL': 1, 'SE_R': False}
regnet_6400M_config = {'WA': 60.83, 'W0': 184, 'WM': 2.07, 'DEPTH': 17, 'GROUP_W': 56, 'BOT_MUL': 1, 'SE_R': False}
regnetY_600M_config = {'WA': 32.54, 'W0': 48, 'WM': 2.32, 'DEPTH': 15, 'GROUP_W': 16, 'BOT_MUL': 1, 'SE_R': 0.25}

def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))       # ks = [0,1,2...,3...]
    ws = w_0 * np.power(w_m, ks)                             # float channel for 4 stages
    ws = np.round(np.divide(ws, q)) * q                      # make it divisible by 8
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    # ws: width list, num_stages: 4, max_stage: 4.0, wscont: float before round width
    return ws, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model."""

    def __init__(self, cfg, **kwargs):
        # Generate RegNet ws per block
        b_ws, num_s, _, _ = generate_regnet(
            cfg['WA'], cfg['W0'], cfg['WM'], cfg['DEPTH']
        )
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [cfg['GROUP_W'] for _ in range(num_s)]
        bms = [cfg['BOT_MUL'] for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage, stride set to 2
        ss = [2 for _ in range(num_s)]
        # Use SE for RegNetY
        se_r = cfg['SE_R']
        # Construct the model
        STEM_W = 32
        kwargs = {
            "stem_w": STEM_W,
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
            "nc": 1000,
        }
        super(RegNet, self).__init__(**kwargs)
