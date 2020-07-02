import matplotlib.pyplot as plt
import numpy as np
import json
from typing import Sequence, Dict, Any
from collections import defaultdict

def get_nas_records(json_file_path: str):
    return json.load(open(json_file_path, 'r'))['data']

_records_T = Sequence[Dict[str, Any]]

def _print_best(records: _records_T):
    sorted_records = sorted(records, key=lambda x: x['mAP'])
    print(sorted_records[-1])

def compute_auc(X, Y):
    X = np.concatenate(([0.], X, [1.]))
    Y = np.concatenate(([0.], Y))
    return np.sum(np.diff(X) * Y)

def compute_EDF(records: _records_T):
    maps = [record['mAP'] for record in records]
    errors = 1 - np.array(maps)
    sorted_errors = np.sort(errors)
    acc_probs = np.arange(1, len(errors) + 1) / len(errors)
    return sorted_errors, acc_probs

def draw_EDF(records: _records_T):
    records = list(filter(_time_filter(0, 60), records))
    print(f'samples: {len(records)}')
    errors, acc_probs = compute_EDF(records)
    plt.plot(errors, acc_probs)
    plt.savefig('results/EDF.png')

def compare_EDF(records: _records_T):
    records = list(filter(_time_filter(45, 65), records))
    cf = _channel_filter((56, 80, 32), (1024, 1024, 768))
    frecords = list(filter(cf, records))
    frecords = list(filter(_d_filter(1, 6), frecords))
    frecords = list(filter(_bm_filter(0.3, 3), frecords))
    print(f'samples: {len(frecords)}')
    xy = compute_EDF(frecords)
    fauc = compute_auc(*xy)
    plt.plot(*xy, color='g', marker='*')
    # srecords = list(filter(_reverse_filter(cf), records))
    srecords = records
    # _print_best(srecords)
    print(f'samples: {len(srecords)}')
    xy = compute_EDF(srecords)
    sauc = compute_auc(*xy)
    print(fauc - sauc)
    plt.plot(*xy, color='r', marker='*')
    plt.savefig('results/compare_EDF.png')

def search_channels(records: _records_T):

    def gen_ch():
        for c1 in range(32, 513, 24):
            for c2 in range(c1 * 2, 1025, 24):
                yield c1, c2

    def gen_triple_ch():
        for p1 in gen_ch():
            for p2 in gen_ch():
                for p3 in gen_ch():
                    yield tuple(zip(p1, p2, p3))

    def compute_auc_diff(channel_setting):
        cf = _channel_filter(*channel_setting)
        frecords = list(filter(cf, records))
        records_limit = len(records) / 3
        if len(frecords) < records_limit or len(frecords) > records_limit * 2:
            return None
        xy = compute_EDF(frecords)
        fauc = compute_auc(*xy)
        srecords = list(filter(_reverse_filter(cf), records))
        xy = compute_EDF(srecords)
        sauc = compute_auc(*xy)
        return fauc - sauc

    records = list(filter(_time_filter(0, 70), records))
    results = {}
    best_auc = 0
    best_set = None
    for i, cs in enumerate(gen_triple_ch()):
        auc_diff = compute_auc_diff(cs)
        if auc_diff is not None and auc_diff > 0:
            results[cs] = auc_diff
            if auc_diff > best_auc:
                best_auc = auc_diff
                best_set = cs
        if i % 100000 == 0:
            print(f'{i}-best: {best_auc}, set: {best_set}')
    print(len(results))
    print(best_set)

def _d_filter(mi, ma):
    def fltr(r):
        if mi <= r['cfg'][0]['d'] <= ma:
            return True
        return False
    return fltr

def _bm_filter(mi, ma):
    def fltr(r):
        if mi <= r['cfg'][0]['bm'] <= ma:
            return True
        return False
    return fltr

def _macs_filter(mi, ma):
    def fltr(r):
        if mi < r['MACs'] < ma:
            return True
        return False
    return fltr

def _time_filter(mi, ma):
    def fltr(r):
        if mi < r['avg_time'] < ma:
            return True
        return False
    return fltr

def _to_list(x):
    try:
        iter(x)
    except TypeError:
        return [x, x, x]
    else:
        return x

def _reverse_filter(f):
    return lambda r: not f(r)

def _channel_filter(mi, ma, reverse=False):
    mi, ma = _to_list(mi), _to_list(ma)
    def fltr(r):
        for i in range(3):
            if not(mi[i] <= r['cfg'][i]['w_out'] <= ma[i]):
                return False
        return True
    if reverse:
        return _reverse_filter(fltr)
    return fltr

def analyze_bm(records: _records_T):
    records = list(filter(_time_filter(45, 65), records))
    print(f'samples: {len(records)}')
    maps = [record['mAP'] for record in records]
    bm = [record['cfg'][0]['bm'] for record in records]
    plt.scatter(bm, maps)
    plt.savefig('results/bm.png')

def analyze_d(records: _records_T):
    records = list(filter(_time_filter(0, 50), records))
    print(f'samples: {len(records)}')
    maps = [record['mAP'] for record in records]
    d = [record['cfg'][0]['d'] for record in records]
    plt.scatter(d, maps)
    plt.savefig('results/d.png')

def analyze_p(records: _records_T):
    records = list(filter(_time_filter(0, 60), records))
    print(f'samples: {len(records)}')
    maps = [record['mAP'] for record in records]
    rp = [record['cfg'][0]['p']/record['cfg'][0]['d'] for record in records]
    plt.scatter(rp, maps)
    plt.savefig('results/rp.png')

def analyze_channel_trend(records: _records_T):
    records = list(filter(_time_filter(0, 80), records))
    print(f'samples: {len(records)}')
    sorted_records = sorted(records, key=lambda x: x['mAP'])
    top_rcrds = sorted_records[len(sorted_records)-5:]
    maps = [record['mAP'] for record in top_rcrds]
    print(maps)
    channels = [[record['cfg'][i]['w_out'] for i in range(3)] for record in top_rcrds]
    for c in channels:
        plt.plot(np.arange(3), c, color='g')
    top_rcrds = sorted_records[:5]
    maps = [record['mAP'] for record in top_rcrds]
    print(maps)
    channels = [[record['cfg'][i]['w_out'] for i in range(3)] for record in top_rcrds]
    for c in channels:
        plt.plot(np.arange(3), c, color='r')
    plt.savefig('results/channel_trend.png')

def analyze_macs_time(records: _records_T):
    gr = defaultdict(list)
    # for r in records:
    #     gw = r['cfg'][0]['gw']
    #     if gw < 2:
    #         label = '<2'
    #     elif 2 < gw < 6:
    #         label = '2~6'
    #     elif 6 < gw < 17:
    #         label = '6~17'
    #     elif 17 < gw < 30:
    #         label = '17~30'
    #     else:
    #         label = '>30'
    #     gr[label].append(r)
    for r in records:
        gr[r['cfg'][0]['d']].append(r)
    for gw, rcd in gr.items():
        macs = [record['MACs'] for record in rcd]
        times = [record['avg_time'] for record in rcd]
        plt.scatter(macs, times, color=tuple(np.random.uniform(size=(3,))), label=str(gw))
    plt.legend()
    plt.savefig('results/macs_vs_time.png')

def analyze_map(records: _records_T):
    macs = [record['MACs'] for record in records]
    times = [record['avg_time'] for record in records]
    maps = [record['mAP'] for record in records]
    against = 'times'
    if against == 'times':
        plt.scatter(times, maps, s=10)
        plt.savefig('results/maps_vs_times.png')
    else:
        plt.scatter(macs, maps, s=10)
        plt.savefig('results/maps_vs_macs.png')

if __name__ == "__main__":
    rcrds = get_nas_records('log/nas.json')
    # draw_EDF(rcrds)
    # analyze_bm(rcrds)
    # analyze_channel_trend(rcrds)
    # analyze_macs_time(rcrds)
    # analyze_d(rcrds)
    # analyze_p(rcrds)
    # compare_EDF(rcrds)
    # search_channels(rcrds)
    analyze_map(rcrds)
