import pytorch_lightning as pl
from torch import nn, zeros, optim, device, cat, tensor


class LSTM(pl.LightningModule):

    def __init__(self, input_size=5, output_size=1, hidden_layer_size=100, look_ahead=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.look_ahead = look_ahead
        self.hidden_layer_size = hidden_layer_size
        self.encoder = nn.LSTM(input_size, hidden_layer_size)
        self.comp = nn.Linear(35, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, 5)
        self.decoder = nn.LSTM(hidden_layer_size, hidden_layer_size)
        self.register_buffer("hidden_cell_1", zeros(1, 1, self.hidden_layer_size, device=device('cuda')))
        self.register_buffer("hidden_cell_2", zeros(1, 1, self.hidden_layer_size, device=device('cuda')))
        self.evaluation_data = []
        self.loss_function = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, batch):
        self.init_hidden()
        lstm_input = []
        # for i in range(0, 5):
        #     lstm_input.append(self.comp(batch[i].view(1, 1, 35)[0][0]))
        encoder_outputs, hidden_cell = self.encoder(batch.view(5, 1, 5),
                                                    (self.hidden_cell_1, self.hidden_cell_2))
        self.hidden_cell_1 = hidden_cell[0]
        self.hidden_cell_2 = hidden_cell[0]
        predictions = []
        lstm_input = encoder_outputs[-1]
        for i in range(0, 5):
            lstm_out, hidden_cell = self.decoder(lstm_input.view(1, len(lstm_input), -1),
                                                 (self.hidden_cell_1, self.hidden_cell_2))
            self.hidden_cell_1 = hidden_cell[0]
            self.hidden_cell_2 = hidden_cell[0]
            lstm_input = lstm_out
            predictions.append(self.linear(lstm_out))
        return cat(predictions, 2)[0][0]

    def training_step(self, batch, batch_idx):
        self.init_hidden()
        predictions = self(batch)
        loss = self.loss_function(predictions.view(1, 5, 5), batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.init_hidden()
        predictions = self.forward(batch)
        loss = self.loss_function(predictions.view(1, 5, 5), batch)

        # return loss
        self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        self.init_hidden()

        predictions = self.forward(batch)
        loss = self.loss_function(tensor(
            [predictions[0].item(), predictions[5].item(), predictions[10].item(), predictions[15].item(),
             predictions[20].item()], device=device('cuda')),
            tensor([item[0] for item in batch], device=device('cuda')))
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # if z.item() > 18:
        #     self.evaluation_data.append(
        #         f'{str(x)} - {str(y)} - {str(predictions)} - {str(z.item())} - {str(z.item() - 13.89)} - {str(loss.item())}')
        #     print(z.item(), loss.item())
        self.evaluation_data.append(
            (loss.item() * 100000, 1 if max([i.item() for i in batch.view(1, 1, 25)[0][0]]) * 20 > 17 else 0))

    def init_hidden(self):
        self.hidden_cell_1 = zeros(1, 1, self.hidden_layer_size, device=device('cuda'))
        self.hidden_cell_2 = zeros(1, 1, self.hidden_layer_size, device=device('cuda'))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
