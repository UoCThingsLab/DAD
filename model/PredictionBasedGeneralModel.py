from torch import nn, zeros, device
from model.trash.LSTM import LSTM


class PredictionBasedGeneralModel(LSTM):
    def __init__(self, hidden_layer_size=100, battle_neck=25, feature_len=3, observe_len=5, label_len=1,
                 objects_len=5,
                 d=device('cuda'), *args,
                 **kwargs):
        super().__init__(hidden_layer_size, device, *args, **kwargs)
        self.LSTM1 = nn.LSTM(objects_len * feature_len, hidden_layer_size)
        self.encoder1 = nn.Linear(hidden_layer_size, battle_neck)
        self.LSTM2 = nn.LSTM(battle_neck, hidden_layer_size)
        self.encoder2 = nn.Linear(hidden_layer_size, objects_len)
        self.battle_neck = battle_neck
        self.feature_len = feature_len
        self.hidden_layer_size = hidden_layer_size
        self.d = d
        self.observe_len = observe_len
        self.label_len = label_len
        self.objects_len = objects_len

    def forward(self, batch):
        # LSTM 1
        h1, h2 = self.init_hidden()
        encoder_outputs, (_, _) = self.LSTM1(batch.view(self.observe_len, 1, self.feature_len * self.objects_len),
                                             (h1, h2))

        # Encoder 1
        encoded = self.encoder1(encoder_outputs[-1])
        encoded = encoded.view(1, 1, self.battle_neck)

        # LSTM 2
        h1, h2 = self.init_hidden()
        output = []
        for i in range(0, self.label_len):
            LSTM_output, (h1, h2) = self.LSTM2(encoded.view(1, 1, self.battle_neck), (h1, h2))
            LSTM_output = LSTM_output[-1].view(1, 1, self.hidden_layer_size)
            # Encode 2
            decoded = self.encoder2(LSTM_output)
            output.append(decoded)
        return output

    def get_loss(self, x, y):
        x = x[0][0][0]
        y = y[1][0][0]
        return self.loss_function(x, y)

    def training_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = self.get_loss(output, batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch[0])
        loss = self.get_loss(output, batch)
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch[0])
        for i in range(0, 5):
            speed = batch[1][0][0][i]
            loss = self.loss_function(output[0][0][0][i], speed)
            # for l in range(0, 5):
            #     speed += math.pow((batch[1].view(1, 5)[0][i].item() * 20 - batch[1].view(1, 5)[0][l].item() * 20), 2)
            # speed = math.sqrt(speed)
            # speed /= 4
            self.evaluation_data.append((loss.item(), ((speed * (batch[3] - batch[4])) + batch[4] - batch[2]).item()))
