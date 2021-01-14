import os

from pytorch_lightning.callbacks import Callback
# from evaluation.validator import Validator


class LSTMCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        f = open(os.path.realpath('.') + '/evaluation.log', 'w')
        f2 = open(os.path.realpath('.') + '/evaluation-ab.log', 'w')
        for ev in pl_module.evaluation_data:
            f.write(str(ev[2] * 100) + ',' + str(ev[1]) + '\n')
            # if ev[0] * 30 - 13.89 > 5.55556:
            if ev[2]*100 < 7.5:
                f2.write(str(ev[2] * 100) + ',' + str(ev[1]) + '\n')
        # Validator(pl_module.evaluation_data, ['speed'])
