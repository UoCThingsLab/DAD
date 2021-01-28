import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from data_module.dataset import DrivingDataMadule
from model.LSTMEncoder import LSTMEncoder
from model.LSTMSeq2Seq import LSTMSeq2Seq
from model.EncoderLSTMEncoder import EncoderLSTMEncoder
from model.Siamese import Siamese
from model.LSTMEncoderLSTM import LSTMEncoderLSTM
from model.LSTMAutoencoder import LSTMAutoencoder
from callbacks.lstm_callback import LSTMCallback
from pytorch_lightning.callbacks import ModelCheckpoint


def main(hparams):
    datamodule = DrivingDataMadule('v0.4', 1000, 100, 5000)

    test = True
    if test:
        model = LSTMEncoderLSTM.load_from_checkpoint(
            "/home/sepehr/PycharmProjects/DAD/lightning_logs/version_56/checkpoints/LSTMEncoderLSTM--v_num=00-epoch=99-validation_loss=0.00-train_loss=0.00.ckpt"
        )
    else:
        model = LSTMEncoderLSTM()

    checkpoint_callback = ModelCheckpoint(
        filename='LSTMEncoderLSTM--{v_num:02d}-{epoch:02d}-{validation_loss:.2f}-{train_loss:.2f}',
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
