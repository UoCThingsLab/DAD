import os

import pytorch_lightning as pl
from torch import tensor, device
from pytorch_lightning.metrics.functional.classification import stat_scores
import matplotlib.pyplot as plt
from model import LSTM


class Validator:
    def __init__(self, data):
        self.test(data)

    # self.accuracy = Accuracy(num_classes=2).to(device("cuda", 0))
    # self.F1 = F1(num_classes=2)
    # self.Precision = Precision(num_classes=2).to(device("cuda", 0))
    # self.Recall = Recall(num_classes=2).to(device("cuda", 0))
    # self.ROC = classification.ROC(pos_label=1)
    # self.ROC = classification.ROC(pos_label=1)

    def test(self, data):
        error = 10;
        tpr = []
        fpr = []
        while error >= 0:
            pred = []
            y = []
            for record in data:
                d, e = record
                pred.append(1 if d > error else 0)
                y.append(e)
            error -= 0.001
            tps, fps, tns, fns, sups = stat_scores(tensor(pred, device=device("cuda", 0)),
                                               tensor(y, device=device("cuda", 0)), class_index=1)
            print(error, tps.item(), fps.item(), tns.item(), fns.item(), tps.item() / (tps.item() + fns.item()),
                  fps.item() / (fps.item() + tns.item()))
            tpr.append(tps.item() / (tps.item() + fns.item()))
            fpr.append(fps.item() / (fps.item() + tns.item()))
        plt.plot(fpr, tpr)

        plt.grid(True)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()


data = []
f = open(os.path.realpath('..') + '/evaluation.log', 'r')
for l in f.readlines():
    data.append((float(l.split(',')[0]), float(l.split(',')[1])))
Validator(data)
