from torch import nn, zeros, optim, device, cat, tensor, mean
from model.LSTM import LSTM


class LSTMEncoder(LSTM):
    def __init__(self, input_size=5, hidden_layer_size=100, *args, **kwargs):
        super().__init__(hidden_layer_size, *args, **kwargs)
        self.encoder = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, input_size * 1)
        self.register_buffer("hidden_cell_1", zeros(1, 1, self.hidden_layer_size, device=device('cuda')))
        self.register_buffer("hidden_cell_2", zeros(1, 1, self.hidden_layer_size, device=device('cuda')))

    def forward(self, batch):
        self.init_hidden()
        encoder_outputs, hidden_cell = self.encoder(batch.view(5, 1, 5),
                                                    (self.hidden_cell_1, self.hidden_cell_2))
        output = self.linear(encoder_outputs[-1])
        return output

    def training_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = self.loss_function(output[0], batch[1].view(1, 5)[0])
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = self.loss_function(output[0], batch[1].view(1, 5)[0])
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = self.loss_function(output[0], batch[1].view(1, 5)[0])
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.evaluation_data.append((loss.item(), 1 if batch[2] > 14 else 0))
