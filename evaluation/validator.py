import os

import pytorch_lightning as pl
from torch import tensor, device
from pytorch_lightning.metrics.functional.classification import stat_scores
import matplotlib.pyplot as plt
from model import LSTMEncoder


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
        error = 1;
        tpr = []
        fpr = []
        while error >= 0:
            pred = []
            y = []
            for record in data:
                d, e = record
                pred.append(1 if d > error else 0)
                y.append(e)
            error -= 0.01
            tps, fps, tns, fns, sups = stat_scores(tensor(pred, device=device("cuda", 0)),
                                                   tensor(y, device=device("cuda", 0)), class_index=1)
            tps = tps.item()
            fps = fps.item()
            tns = tns.item()
            fns = fns.item()
            pres = 0
            if tps + fps > 0:
                pres = tps / (tps + fps)
            rec = 0
            if tps + fns > 0:
                rec = tps / (tps + fns)
            f1 = 0
            if rec + pres > 0:
                f1 = (2 * pres * rec) / (pres + rec)
            acc = (tps + tns) / (tps + fns + fps + tns)
            tprate = tps / (tps + fns)
            fprate = fps / (fps + tns)
            # print(error, tps.item(), fps.item(), tns.item(), fns.item(), tps.item() / (tps.item() + fns.item()),
            #       fps.item() / (fps.item() + tns.item()))
            print(
                f'Error: {error} tp:{tps} fp:{fps} tn:{tns} fn:{fns}\n'
                + f'TP Rate: {tprate * 100}% FP Rate: {fprate * 100}%'
                + f'Accuracy: {acc * 100}% Precision:{pres * 100}% Recall:{rec * 100}% F1:{f1 * 100}%'
                  f'\n-------------------------------------------------------------------------')
            tpr.append(tprate)
            fpr.append(fprate)
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
