from torch import nn, zeros, device, cat
from model.trash.LSTM import LSTM


class LSTMAutoencoder(LSTM):
    def __init__(self, input_size=5, hidden_layer_size=100, bottle_neck=3, *args, **kwargs):
        super().__init__(hidden_layer_size, *args, **kwargs)
        self.encoder1 = nn.LSTM(input_size, hidden_layer_size)
        self.encoder2 = nn.LSTM(hidden_layer_size, bottle_neck)
        self.decoder1 = nn.LSTM(bottle_neck, hidden_layer_size)
        self.decoder2 = nn.LSTM(hidden_layer_size, input_size)
        self.register_buffer("hidden_cell_1", zeros(1, 1, self.hidden_layer_size, device=device('cuda')))
        self.register_buffer("hidden_cell_2", zeros(1, 1, self.hidden_layer_size, device=device('cuda')))
        self.init_hidden()

    def forward(self, batch):
        encoder_outputs, (_, _) = self.encoder1(batch.view(5, 1, 5),
                                                (self.hidden_cell_1, self.hidden_cell_2))
        encoder2_outputs = []
        encoder_outputs = encoder_outputs[-1]
        self.hidden_cell_1, self.hidden_cell_2 = zeros(1, 1, 3, device=device('cuda')), zeros(1, 1, 3,
                                                                                              device=device('cuda'))
        for i in range(0, 5):
            encoder2_output, (self.hidden_cell_1, self.hidden_cell_2) = \
                self.encoder2(encoder_outputs.view(1, 1, 100), (self.hidden_cell_1, self.hidden_cell_2))
            encoder2_outputs.append(encoder2_output[-1])
        encoder2_outputs = cat(encoder2_outputs)
        self.hidden_cell_1, self.hidden_cell_2 = zeros(1, 1, 100, device=device('cuda')), zeros(1, 1, 100,
                                                                                                device=device('cuda'))
        decoder_outputs, (_, _) = self.decoder1(encoder2_outputs.view(5, 1, 3),
                                                (self.hidden_cell_1, self.hidden_cell_2))
        decoder2_outputs = []
        decoder_outputs = decoder_outputs[-1]
        self.hidden_cell_1, self.hidden_cell_2 = zeros(1, 1, 5, device=device('cuda')), zeros(1, 1, 5,
                                                                                              device=device('cuda'))
        for i in range(0, 5):
            decoder2_output, (self.hidden_cell_1, self.hidden_cell_2) = \
                self.decoder2(decoder_outputs.view(1, 1, 100), (self.hidden_cell_1, self.hidden_cell_2))
            decoder2_outputs.append(decoder2_output[-1])
        self.init_hidden()
        return cat(decoder2_outputs).view(5, 1, 5)

    def training_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = self.loss_function(output.view(25), batch[0].view(25))
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = self.loss_function(output.view(25), batch[0].view(25))
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = self.loss_function(output[0], batch[1].view(1, 5)[0])
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.evaluation_data.append((loss.item(), 1 if batch[2] > 14 else 0))
