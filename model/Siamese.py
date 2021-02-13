from torch import nn, zeros, device, cat, tensor
from model.trash.LSTM import LSTM


class Siamese(LSTM):
    def __init__(self, hidden_layer_size=100, battle_neck=12, feature_len=6, observe_len=5, label_len=1,
                 objects_len=5,
                 d=device('cuda'), *args,
                 **kwargs):
        super().__init__(hidden_layer_size, device, *args, **kwargs)
        self.LSTM1 = nn.LSTM(feature_len, hidden_layer_size)
        self.encoder1 = nn.Linear(hidden_layer_size, battle_neck)
        self.LSTM2 = nn.LSTM(battle_neck, hidden_layer_size)
        self.encoder2 = nn.Linear(hidden_layer_size, feature_len)
        self.battle_neck = battle_neck
        self.feature_len = feature_len
        self.hidden_layer_size = hidden_layer_size
        self.d = d
        self.observe_len = observe_len
        self.label_len = label_len
        self.objects_len = objects_len
        self.similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, batch):
        # LSTM 1
        h1, h2 = self.init_hidden()
        encoder_outputs, (_, _) = self.LSTM1(batch.view(self.observe_len, 1, self.feature_len),
                                             (h1, h2))

        # Encoder 1
        encoded = self.encoder1(encoder_outputs[-1])
        encoded = encoded.view(1, 1, self.battle_neck)
        # LSTM 2
        h1, h2 = self.init_hidden()
        output = []
        for i in range(0, self.observe_len):
            LSTM_output, (h1, h2) = self.LSTM2(encoded, (h1, h2))
            LSTM_output = LSTM_output[-1].view(1, 1, self.hidden_layer_size)
            # Encode 2
            decoded = self.encoder2(LSTM_output)
            output.append(decoded)
        return output, encoded

    def get_loss(self, x, y):
        return self.loss_function(x, y)

    def training_step(self, batch, batch_idx):
        loss = 0
        decoded = []
        for j in range(0, self.objects_len):
            list = [[batch[0][0][i][k][j].item() for k in range(0, self.feature_len)] for i in
                    range(0, self.observe_len)]
            list = tensor(list, device=self.d)
            output, d = self(list)
            decoded.append(d[0][0])
            output = cat(output).view(self.feature_len * self.observe_len)
            list = list.view(self.feature_len * self.observe_len)
            loss += self.get_loss(output, list)
        loss2 = 0
        for i in range(0, self.objects_len):
            for j in range(0, self.objects_len):
                if i != j:
                    l = self.similarity(decoded[i], decoded[j])
                    l = (-1 * l) + 1
                    loss2 += l
        loss = loss2 + (10 * loss)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = 0
        decoded = []
        for j in range(0, self.objects_len):
            list = [[batch[0][0][i][k][j].item() for k in range(0, self.feature_len)] for i in
                    range(0, self.observe_len)]
            list = tensor(list, device=self.d)
            output, d = self(list)
            decoded.append(d[0][0])
            output = cat(output).view(self.feature_len * self.observe_len)
            list = list.view(self.feature_len * self.observe_len)
            loss += self.get_loss(output, list)
        loss2 = 0
        for i in range(0, self.objects_len):
            for j in range(0, self.objects_len):
                if i != j:
                    l = self.similarity(decoded[i], decoded[j])
                    l = (-1 * l) + 1
                    loss2 += l
        loss = loss2 + (10 * loss)
        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        decoded = []
        maxSpeed = []

        for j in range(0, self.objects_len):
            list = [[batch[0][0][i][k][j].item() for k in range(0, self.feature_len)] for i in
                    range(0, self.observe_len)]
            list = tensor(list, device=self.d)
            o, d = self(list)
            m = max(batch[1][0][j]).item()
            maxSpeed.append(m)
            decoded.append(d)
        for i in range(0, self.objects_len):
            loss = 0
            for j in range(0, self.objects_len):
                if i != j:
                    l = self.similarity(decoded[i][0][0], decoded[j][0][0])
                    loss += (-1 * l) + 1
            loss /= 4
            self.evaluation_data.append(
                (loss.item(), ((maxSpeed[i] * (batch[3] - batch[4])) + batch[4] - batch[2]).item()))
            # self.evaluation_data.append((loss.item(), 1 if batch[2].item() == i else 0))
