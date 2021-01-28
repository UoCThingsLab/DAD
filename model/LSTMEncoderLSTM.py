from torch import nn, zeros, optim, device, cat, tensor, mean
from model.LSTM import LSTM
import math


class LSTMEncoderLSTM(LSTM):
    def __init__(self, input_size=5, hidden_layer_size=100, battle_neck=15, *args, **kwargs):
        super().__init__(hidden_layer_size, *args, **kwargs)
        self.LSTM1 = nn.LSTM(input_size, hidden_layer_size)
        self.encoder1 = nn.Linear(300, battle_neck)
        self.LSTM2 = nn.LSTM(battle_neck, hidden_layer_size)
        self.encoder2 = nn.Linear(hidden_layer_size, input_size)

    def forward(self, batch):
        hidden_cell_1 = zeros(1, 3, self.hidden_layer_size, device=device('cuda'))
        hidden_cell_2 = zeros(1, 3, self.hidden_layer_size, device=device('cuda'))
        encoder_outputs, (_, _) = self.LSTM1(batch[0],
                                             (hidden_cell_1, hidden_cell_2))
        encoded = self.encoder1(encoder_outputs[-1].view(300))
        hidden_cell_1 = zeros(1, 1, self.hidden_layer_size, device=device('cuda'))
        hidden_cell_2 = zeros(1, 1, self.hidden_layer_size, device=device('cuda'))
        decoder_outputs, (_, _) = self.LSTM2(encoded.view(1, 1, 15),
                                             (hidden_cell_1, hidden_cell_2))
        decoded = self.encoder2(decoder_outputs[-1].view(1, 1, 100))
        return decoded

    def training_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = self.loss_function(output[0][0], batch[1].view(1, 5)[0])
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = self.loss_function(output[0][0], batch[1].view(1, 5)[0])
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        output = self([batch[0]])
        for i in range(0, 5):
            loss = self.loss_function(output[0][0][i], batch[1].view(1, 5)[0][i])
            speed = batch[1].view(1, 5)[0][i]
            # for l in range(0, 5):
            #     speed += math.pow((batch[1].view(1, 5)[0][i].item() * 20 - batch[1].view(1, 5)[0][l].item() * 20), 2)
            # speed = math.sqrt(speed)
            # speed /= 4
            self.evaluation_data.append((loss.item(), (speed * 20 - batch[2]).item()))
