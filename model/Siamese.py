from torch import nn, zeros, optim, device, cat, tensor, mean, max
from model.LSTM import LSTM


class Siamese(LSTM):
    def __init__(self, input_size=1, hidden_layer_size=100, bottle_neck=15, *args, **kwargs):
        super().__init__(hidden_layer_size, *args, **kwargs)
        self.LSTM1 = nn.LSTM(input_size, hidden_layer_size)
        self.endocer = nn.Linear(hidden_layer_size, bottle_neck)
        self.decoder = nn.Linear(bottle_neck, hidden_layer_size)
        self.LSTM2 = nn.LSTM(hidden_layer_size, hidden_layer_size)
        self.decoder2 = nn.Linear(hidden_layer_size, input_size)
        self.register_buffer("hidden_cell_1", zeros(1, 1, self.hidden_layer_size, device=device('cuda')))
        self.register_buffer("hidden_cell_2", zeros(1, 1, self.hidden_layer_size, device=device('cuda')))
        self.init_hidden()

    def forward(self, batch):
        output = []
        for i in range(0, 5):
            self.init_hidden()
            input = tensor([batch.view(25)[5 * j + i] for j in range(0, 5)], device=device('cuda'))
            LSTM1_outputs, (_, _) = self.LSTM1(input.view(5, 1, 1),
                                               (self.hidden_cell_1, self.hidden_cell_2))
            encoded = self.endocer(LSTM1_outputs[-1])
            LSTM_input = self.decoder(encoded)
            LSTM2_outputs = []
            for i in range(0, 5):
                LSTM2_output, (self.hidden_cell_1, self.hidden_cell_2) = \
                    self.LSTM2(LSTM_input.view(1, 1, 100), (self.hidden_cell_1, self.hidden_cell_2))
                LSTM_input = LSTM2_output[-1]
                LSTM2_outputs.append(self.decoder2(LSTM2_output))
            output.append(cat(LSTM2_outputs))

        return [cat([output[i][j][0] for j in range(0, 5)]) for i in range(0, 5)]

    def training_step(self, batch, batch_idx):
        output = self(batch[0])
        loss1 = 0
        loss2 = 0
        inputs = []
        for i in range(0, 5):
            input = tensor([batch[0].view(25)[5 * j + i] for j in range(0, 5)], device=device('cuda'))
            loss1 += self.loss_function(output[i], input)
            inputs.append(input)
        for i in range(0, 5):
            for j in range(0, 5):
                loss2 += self.loss_function(inputs[i], inputs[j])
        loss = (loss1 + loss2) / 5
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch[0])
        loss1 = 0
        loss2 = 0
        inputs = []
        for i in range(0, 5):
            input = tensor([batch[0].view(25)[5 * j + i] for j in range(0, 5)], device=device('cuda'))
            loss1 += self.loss_function(output[i], input)
            inputs.append(input)
        for i in range(0, 5):
            for j in range(0, 5):
                loss2 += self.loss_function(inputs[i], inputs[j])
        loss = (loss1 + loss2) / 5
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch[0])
        max_e, max_s = 0, 0
        inputs = []
        for i in range(0, 5):
            input = tensor([batch[0].view(25)[5 * j + i] for j in range(0, 5)], device=device('cuda'))
            inputs.append(input)
        for i in range(0, 5):
            for j in range(0, 5):
                if self.loss_function(output[i], output[j]) > max_e:
                    max_e = self.loss_function(output[i], output[j])
                    max_s = max(inputs[i], inputs[j])
        self.log('test_loss', max_e, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.evaluation_data.append((max_e, 1 if max_s * 20 > 14 else 0))
