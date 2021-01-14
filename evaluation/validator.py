import os

import pytorch_lightning as pl
from torch import tensor, device
from pytorch_lightning.metrics.functional.classification import stat_scores
import matplotlib.pyplot as plt
from model import LSTM


class Validator:
    def __init__(self, data, tests):
        # self.accuracy = Accuracy(num_classes=2).to(device("cuda", 0))
        # self.F1 = F1(num_classes=2)
        # self.Precision = Precision(num_classes=2).to(device("cuda", 0))
        # self.Recall = Recall(num_classes=2).to(device("cuda", 0))
        # self.ROC = classification.ROC(pos_label=1)
        # self.ROC = classification.ROC(pos_label=1)
        if 'speed' in tests:
            self.speed_test(data)
        if 'distance' in tests:
            self.distance_test(data)

    def speed_test(self, data):

        speed_limit = 13.89
        speed_thershold = [1.38889, 2.77778, 4.16667, 5.55556, 6.94444]
        label = ['5kph', '10kph', '15kph', '20kph', '25kph']
        for i in range(0, 4):
            error = 1
            st = speed_thershold[i]
            tpr = []
            fpr = []
            while error >= 0:
                pred = []
                y = []
                for record in data:
                    d, e = record
                    pred.append(1 if e > error else 0)
                    y.append(1 if d - speed_limit > st else 0)
                # print('Accuracy',
                #       self.accuracy(tensor(pred, device=device("cuda", 0)), tensor(y, device=device("cuda", 0))).item())
                # print('Precision',
                #       self.Precision(tensor(pred, device=device("cuda", 0)), tensor(y, device=device("cuda", 0))).item())
                # print('Recall',
                #       self.Recall(tensor(pred, device=device("cuda", 0)), tensor(y, device=device("cuda", 0))).item())
                # fpr, tpr, thresholds = self.ROC(tensor(pred, device=device("cuda", 0)),
                #                                 tensor(y, device=device("cuda", 0)))
                tps, fps, tns, fns, sups = stat_scores(tensor(pred, device=device("cuda", 0)),
                                                       tensor(y, device=device("cuda", 0)), class_index=1)
                print(error, tps.item(), fps.item(), tns.item(), fns.item(), tps.item() / (tps.item() + fns.item()),
                      fps.item() / (fps.item() + tns.item()))
                tpr.append(tps.item() / (tps.item() + fns.item()))
                fpr.append(fps.item() / (fps.item() + tns.item()))
                error -= 0.001
            plt.plot(fpr, tpr, label=label[i])

        plt.grid(True)
        plt.legend(label)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        # print('F1', self.F1(tensor(pred), tensor(y)).item())
        # print('Precision', self.Precision(tensor(pred), tensor(y)).item())
        # print('Recall', self.Recall(tensor(pred), tensor(y)).item())

    def distance_test(self,data):
        error = 1;
        tpr = []
        fpr = []
        while error >= 0:
            pred = []
            y = []
            for record in data:
                d, e = record
                pred.append(1 if e > error else 0)
                y.append(1 if d < 7.5 else 0)
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
Validator(data, ['distance'])
