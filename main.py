import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from data_module.dataset import DrivingDataMadule
from model.Siamese import Siamese
from model.callbacks.lstm_callback import LSTMCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main(hparams):
    datamodule = DrivingDataMadule('v0.1', 5800, 176, 100)

    test = False

    if test:
        model = Siamese.load_from_checkpoint(
            "lightning_logs/version_40/checkpoints/LSTMEncoderLSTM--v_num=00-epoch=03-validation_loss=-0.00000-train_loss=0.00000.ckpt"
        )
    else:
        model = Siamese()

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        filename='LSTMEncoderLSTM--{v_num:02d}-{epoch:02d}-{validation_loss:.5f}-{train_loss:.5f}',
    )
    early_callback = EarlyStopping(monitor='validation_loss')

    trainer = pl.Trainer(gpus=-1, max_epochs=100, accelerator='dp',
                         callbacks=[LSTMCallback(), checkpoint_callback],
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
