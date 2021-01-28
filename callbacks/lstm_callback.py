import os

from pytorch_lightning.callbacks import Callback


# from evaluation.validator import Validator


class LSTMCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        f = open(os.path.realpath('.') + '/evaluation.log', 'w')
        f2 = open(os.path.realpath('.') + '/evaluation-ab.log', 'w')
        f3 = open(os.path.realpath('.') + '/evaluation-ab2.log', 'w')
        for ev in pl_module.evaluation_data:
            f.write(str(ev[0]) + ',' + str(ev[1]) + '\n')
            if ev[1] < 0:
                f2.write(str(ev[0]) + ',' + str(ev[1]) + '\n')
            if ev[0] > 0.0017999999:
                f3.write(str(ev[0]) + ',' + str(ev[1]) + '\n')
        # Validator(pl_module.evaluation_data, ['speed'])
