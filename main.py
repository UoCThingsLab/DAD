import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from data_module.dataset import DrivingDataMadule
from model.LSTMEncoder import LSTMEncoder
from callbacks.lstm_callback import LSTMCallback
from pytorch_lightning.callbacks import ModelCheckpoint


def main(hparams):
    datamodule = DrivingDataMadule('v0.3', 10000, 100, 5000)

    test = False
    if test:
        model = LSTMEncoder.load_from_checkpoint(
            "/home/sepehr/PycharmProjects/DAD/lightning_logs/version_81/checkpoints/epoch=44-step=440956.ckpt"
        )
    else:
        model = LSTMEncoder()

    checkpoint_callback = ModelCheckpoint(
        filename='LSTMEncoder--{v_num:02d}-{epoch:02d}-{validation_loss:.2f}-{train_loss:.2f}',
    )
    trainer = pl.Trainer(gpus=-1, max_epochs=100, accelerator='dp', callbacks=[LSTMCallback(), checkpoint_callback],
                         num_nodes=1)
    if test:
        trainer.test(model=model, test_dataloaders=datamodule.test_dataloader())
    else:
        trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    hyperparams = parent_parser.parse_args()

    # TRAIN
    main(hyperparams)
