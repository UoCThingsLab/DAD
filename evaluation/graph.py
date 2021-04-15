import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import tensor
from pytorch_lightning.metrics.functional.classification import stat_scores
from pytorch_lightning import metrics

result_map = {
    "1": "Bottleneck = 8",
    # "2": "Bottleneck = 10",
    "3": "Bottleneck = 12",
    "4": "Bottleneck = 14",
    "5": "Bottleneck = 16",
    "6": "lambda = 1",
    # "2": "lambda = 10",
    "7": "lambda = 0.1",
    "2": "dataset = 0.9",
    "8": "dataset = 0.11",
    "9": "dataset = 0.5",
    "10": "GAK",
    "11": "DTW",
    "12": "iForest"
}

graph = ["4", "11", "10", "12"]

result = []
accuracy = metrics.Accuracy()
F1 = metrics.F1(num_classes=1)
precision = metrics.Precision(num_classes=1)
recall = metrics.Recall(num_classes=1)
all_data = []
for g in graph:
    data = []
    f = open(os.path.realpath('..') + '/result/' + g + '.log', 'r')
    for l in f.readlines():
        data.append((float(l.split(',')[0]), float(l.split(',')[1])))
    error = 3
    tpr = []
    fpr = []
    recs = []
    perss = []
    best = (0, {})
    while error >= -1:
        pred = []
        y = []
        for record in data:
            d, e = record
            try:
                pred.append(1 if d > error else 0)
            except:
                print(d)
            y.append(1 if e else 0)
        error -= 0.01

        pred = tensor(pred)
        y = tensor(y)

        tps, fps, tns, fns, sups = stat_scores(pred, y, class_index=1)
        tps = tps.item()
        fps = fps.item()
        tns = tns.item()
        fns = fns.item()
        rec = recall(pred, y)
        pres = precision(pred, y)
        f1 = F1(pred, y)
        acc = accuracy(pred, y)
        tprate = tps / (tps + fns)
        fprate = fps / (fps + tns)
        # print(error, tps.item(), fps.item(), tns.item(), fns.item(), tps.item() / (tps.item() + fns.item()),
        #       fps.item() / (fps.item() + tns.item()))
        print(
            f'Error: {error} tp:{tps} fp:{fps} tn:{tns} fn:{fns}\n'
            + f'TP Rate: {tprate * 100}% FP Rate: {fprate * 100}%'
            + f'Accuracy: {acc * 100}% Precision:{pres * 100}% Recall:{rec * 100}% F1:{f1 * 100}%'
              f'\n-------------------------------------------------------------------------')
        if f1 > best[0]:
            best = (f1,
                    {"f1": f1, "rec": rec, "pres": pres, "tprate": tprate, "fprate": fprate, "tps": tps, "fps": fps,
                     "tns": tns,
                     "fns": fns, "error": error, "acc": acc
                     })

        tpr.append(tprate)
        fpr.append(fprate)

        if tps == 0 and fps == 0:
            recs.append(0)
            perss.append(1)
        elif rec.item() != 0 or pres.item() != 0:
            recs.append(rec.item())
            perss.append(pres.item())
    all_data.append((tpr, fpr))
# Change the style of plot
plt.style.use('seaborn-darkgrid')

# Create a color palette
palette = plt.get_cmap('Dark2')
num = 0
for num in range(len(all_data)):
    d = all_data[num]
    plt.plot(d[1], d[0], marker='', color=palette(num), linewidth=1, alpha=0.9, label=result_map[graph[num]])
plt.legend(loc=0, ncol=1)
plt.title("A (bad) Spaghetti plot", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Time")
plt.ylabel("Score")
plt.show()
