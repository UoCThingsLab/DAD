import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from data_module.dataset import DrivingDataMadule
from model.LSTM import LSTM
from callbacks.lstm_callback import LSTMCallback


def main(hparams):
    datamodule = DrivingDataMadule('v0.2', 100, 60, 100)

    model = LSTM.load_from_checkpoint(
        "/home/sepehr/PycharmProjects/DAD/lightning_logs/version_2/checkpoints/epoch=90-step=380834.ckpt"
    )
    # model = LSTM()
    trainer = pl.Trainer(gpus=-1, max_epochs=100, accelerator='dp', callbacks=[LSTMCallback()],
                         num_nodes=1)
    # trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, test_dataloaders=datamodule.test_dataloader())


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    hyperparams = parent_parser.parse_args()

    # TRAIN
    main(hyperparams)
