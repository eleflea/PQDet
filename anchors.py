import glob
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric

def iou_distance_wh(wh1, wh2):
    wh1 = wh1 / 2
    wh2 = wh2 / 2
    inter = np.prod(np.minimum(wh1, wh2))
    union = np.prod(wh1) + np.prod(wh2) - inter
    return 1 - inter / (union + 1e-10)

sets=['VisDrone2019-DET-train', 'VisDrone2019-DET-val']
label_files = chain(*[glob.glob('/home/eleflea/dataset/VisDrone2019/%s/annotations/*.txt' % s) for s in sets])
ws = []
hs = []
for file in label_files:
    with open(file, 'r') as fr:
        for line in fr.readlines():
            ann = line.split(',')
            if int(ann[5]) in {0, 11} or int(ann[4]) == 0:
                continue
            w, h = int(ann[2]), int(ann[3])
            ws.append(w)
            hs.append(h)
print(f'{len(ws)} bboxes')

samples = np.array(list(zip(ws, hs)))
for _ in range(1):
    sample = samples[np.random.choice(samples.shape[0], 20000, replace=False), :]
    metric = distance_metric(type_metric.USER_DEFINED, func=iou_distance_wh)
    initial_centers = kmeans_plusplus_initializer(sample, 9).initialize()
    # Create instance of K-Means algorithm with prepared centers.
    kmeans_instance = kmeans(sample, initial_centers, metric=metric)
    # Run cluster analysis and obtain results.
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    final_centers = np.array(kmeans_instance.get_centers())
    # Visualize obtained results
    kmeans_visualizer.show_clusters(sample, clusters, final_centers, display=False)
    sccs = np.round(final_centers[np.argsort(np.prod(final_centers, axis=1))]).astype(np.int)
    print(sccs)
# [[  9  13]
#  [ 25  17]
#  [ 16  31]
#  [ 47  29]
#  [ 32  51]
#  [ 83  48]
#  [ 61  91]
#  [131  99]
#  [210 189]]
plt.savefig('results/visdrone_anchors.png')