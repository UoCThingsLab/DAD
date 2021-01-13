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

    def speed_test(self, data):

        error = 100
        speed_limit = 13.89
        speed_thershold = 5.55556

        pred = []
        y = []
        tpr = []
        fpr = []
        while error >= 0:
            pred = []
            y = []
            for record in data:
                d, e = record
                pred.append(1 if e > error else 0)
                y.append(1 if d - speed_limit > speed_thershold else 0)
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
            error -= 0.25
        plt.plot(fpr, tpr)
        plt.grid(True)
        plt.show()
        # print('F1', self.F1(tensor(pred), tensor(y)).item())
        # print('Precision', self.Precision(tensor(pred), tensor(y)).item())
        # print('Recall', self.Recall(tensor(pred), tensor(y)).item())
