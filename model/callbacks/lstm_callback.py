import os

from pytorch_lightning.callbacks import Callback


# from evaluation.validator import Validator


class LSTMCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        f = open(os.path.realpath('') + '/evaluation.log', 'w')
        f2 = open(os.path.realpath('') + '/evaluation-ab.log', 'w')
        f3 = open(os.path.realpath('') + '/evaluation-ab2.log', 'w')
        for i in range(0, len(pl_module.evaluation_data)):
            ev = pl_module.evaluation_data[i]
            f.write(str(ev[0]) + ',' + str(ev[1]) + '\n')
            # if (i + 1) % 5 == 0:
            #     f.write("\n")
        # Validator(pl_module.evaluation_data, ['speed'])
