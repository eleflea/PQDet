import torch
from torch import nn
from model.interpreter import DetectionModel
import matplotlib.pyplot as plt
import numpy as np
import json

plt.rcParams['font.sans-serif']=['SimHei']

def state_dict_from_path(weight: str):
    state_dict = torch.load(weight, map_location='cpu')
    return state_dict['model']

def model_with_weight(cfg_path: str, weight: str):
    model = nn.DataParallel(DetectionModel(cfg_path))
    state_dict = state_dict_from_path(weight)
    model.load_state_dict(state_dict)
    return model

def draw_bn_scatter(axis, weight, color='b', label=None):
    sd = state_dict_from_path(weight)
    sorted_bns = get_sorted_bn(sd)
    length = len(sorted_bns)
    X = np.arange(0, length) / (length-1)
    axis.scatter(X, sorted_bns, s=25, c=color, alpha=.5, label=label)
    return axis

def draw_multi_bn_scatter(weights, labels=None):
    if labels is None:
        label_texts = [None for _ in range(len(weights))]
    else:
        label_texts = labels
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    assert len(weights) <= len(colors), 'too many weights to draw'
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.xlabel('比例')
    plt.ylabel('参数绝对值大小')
    for weight, color, label in zip(weights, colors, label_texts):
        draw_bn_scatter(ax, weight, color, label)
    if labels is not None:
        plt.legend()
    return fig

def get_sorted_bn(state_dict):
    bns = []
    for key, params in state_dict.items():
        if not(len(params.shape) == 1 and key.endswith('weight')):
            continue
        bns.extend(params.abs().tolist())
    sorted_bns = np.array(bns, dtype=np.float32)
    sorted_bns.sort()
    return sorted_bns

def draw_compare_channel(weight, pruned_weight):
    english = [
        'layers',
        'channels',
        'remaining channels',
        'pruned channels',
    ]
    chinese = [
        '层数',
        '通道数',
        '剪枝后通道数目',
        '剪枝部分',
    ]
    language = english
    ori_chan_nums = get_channel_nums(state_dict_from_path(weight))
    pruned_chan_nums = get_channel_nums(state_dict_from_path(pruned_weight))
    assert len(ori_chan_nums) == len(pruned_chan_nums), 'differnt conv layer amount'
    diff_chan_nums = ori_chan_nums - pruned_chan_nums
    fig, ax = plt.subplots(figsize=(8, 4))
    X = np.arange(1, len(ori_chan_nums)+1)
    plt.xlabel(language[0])
    plt.ylabel(language[1])
    ax.bar(X, pruned_chan_nums, label=language[2])
    ax.bar(X, diff_chan_nums, bottom=pruned_chan_nums, label=language[3])
    plt.legend()
    return fig

def get_channel_nums(state_dict):
    channels = []
    for key, params in state_dict.items():
        if not(len(params.shape) == 4 and key.endswith('weight')):
            continue
        channels.append(params.shape[0])
    channels = np.array(channels, dtype=np.int)
    return channels

def draw_evol(file_path: str):
    records = json.load(open(file_path, 'r'))['data']
    fig, ax = plt.subplots(3, 4, figsize=(8, 6))
    names = records[0]['hyper'].keys()
    for record in records:
        hypers = record['hyper']
        fitness = record['fitness']
        for i, name in enumerate(names):
            ax[i//4, i%4].title.set_text(name)
            ax[i//4, i%4].scatter(hypers[name], fitness, c='b', s=10)

if __name__ == "__main__":
    weights = ['weights/model-74-0.7724.pt',
    'weights/trained/model-74-0.7724.pt',
    'weights/trained/pruned-model-19-0.7458.pt'
    ]
    # labels = ['正常训练', '稀疏训练', '稀疏训练微调后']
    # draw_multi_bn_scatter(weights, labels)
    # draw_evol('evol.json')

    # weights = ['weights/rsod_std_sparse/model-499-0.8978.pt',
    # 'weights/rsod_std_sparse/pruned-85-model-194-0.9081.pt'
    # ]
    draw_compare_channel(*weights[1:])
    plt.savefig('draw.png')