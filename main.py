import pytorch_lightning as pl
from data_module.dataset import DrivingDataMadule
from model.LSTM import LSTM
from callbacks.lstm_callback import LSTMCallback

datamodule = DrivingDataMadule('v0.1', 100, 600, 1000)

# model = LSTM.load_from_checkpoint(
#     "/home/sepehr/PycharmProjects/Neuropad/DAD/model/lightning_logs/version_31/checkpoints/checkpoint.ckpt"
# )
model = LSTM()
trainer = pl.Trainer(gpus=-1, max_epochs=100, accelerator='dp', callbacks=[LSTMCallback()])
trainer.fit(model=model, datamodule=datamodule)
# trainer.test(model=model, test_dataloaders=datamodule.test_dataloader())
