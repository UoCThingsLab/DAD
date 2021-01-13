from pytorch_lightning.callbacks import Callback
from evaluation.validator import Validator


class LSTMCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        f = open('/evaluation.log', 'w')
        for ev in pl_module.evaluation_data:
            f.write(ev + '\n')
        Validator(pl_module.evaluation_data, ['speed'])
