from torch import nn, zeros, optim, device
import pytorch_lightning as pl


class LSTM(pl.LightningModule):
    def __init__(self, hidden_layer_size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.loss_function = nn.MSELoss()
        self.evaluation_data = []

    def init_hidden(self):
        self.hidden_cell_1 = zeros(1, 1, self.hidden_layer_size, device=device('cuda'))
        self.hidden_cell_2 = zeros(1, 1, self.hidden_layer_size, device=device('cuda'))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
