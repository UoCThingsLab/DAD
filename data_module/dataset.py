import os
from torch.utils.data import Dataset
from data_module.XML_parrser import XMLParser
from torch import tensor, device, mean, std
from torch.utils.data import DataLoader
import pytorch_lightning as pl

'''
 Creates and loads the Driving dataset
 address: dataset address
 observe_len: Observation sequence length
 label_leb: Label sequence length
 num_objects: number of cars
'''


class DrivingDataset(Dataset):

    def __init__(self, address, start=0, end=10000, observe_len=10, label_len=1, num_objects=5, d=device('cuda')):
        self.parser = XMLParser(address)
        self.start = start
        self.end = end
        self.observe_len = observe_len
        self.label_len = label_len
        self.num_objects = num_objects
        self.device = d
        self.set = []

        # Params Config
        self.speed_index = 0
        self.min_speed = 0
        self.max_speed = 20
        self.x_index = 1
        self.max_x = 965.07
        self.min_x = -41.12
        self.y_index = 2
        self.max_y = 50.60
        self.min_y = 44.20
        self.l_index = 3
        self.load_set()

    def normalize(self, d, max, min):
        return (d - min) / (max - min)

    def load_set(self):
        raw_data = self.parser.read_txt()
        for round in raw_data:
            for i in range(2, len(round) - self.observe_len - self.label_len + 2):
                seq = []
                max_speed = 0
                min_speed = 10000
                max_x = -10000
                min_x = 10000
                max_y = -10000
                min_y = 10000
                max_d_x = -100000
                min_d_x = 100000
                max_dd_x = -100000
                min_dd_x = 100000
                max_d_y = -100000
                min_d_y = 100000
                max_dd_y = -100000
                min_dd_y = 100000
                for j in range(i, i + self.observe_len):
                    for k in range(0, self.num_objects):
                        if max_speed < round[j][0][k][self.speed_index]:
                            max_speed = round[j][0][k][self.speed_index]
                        if min_speed > round[j][0][k][self.speed_index]:
                            min_speed = round[j][0][k][self.speed_index]

                        if max_x < round[j][0][k][self.x_index]:
                            max_x = round[j][0][k][self.x_index]
                        if min_x > round[j][0][k][self.x_index]:
                            min_x = round[j][0][k][self.x_index]
                        if max_d_x < round[j][0][k][self.x_index] - round[j - 1][0][k][self.x_index]:
                            max_d_x = round[j][0][k][self.x_index] - round[j - 1][0][k][self.x_index]
                        if min_d_x > round[j][0][k][self.x_index] - round[j - 1][0][k][self.x_index]:
                            min_d_x = round[j][0][k][self.x_index] - round[j - 1][0][k][self.x_index]
                        if max_dd_x < (round[j][0][k][self.x_index] - round[j - 1][0][k][self.x_index]) - (
                                round[j - 1][0][k][self.x_index] - round[j - 2][0][k][self.x_index]):
                            max_dd_x = (round[j][0][k][self.x_index] - round[j - 1][0][k][self.x_index]) - (
                                    round[j - 1][0][k][self.x_index] - round[j - 2][0][k][self.x_index])
                        if min_dd_x > (round[j][0][k][self.x_index] - round[j - 1][0][k][self.x_index]) - (
                                round[j - 1][0][k][self.x_index] - round[j - 2][0][k][self.x_index]):
                            min_dd_x = (round[j][0][k][self.x_index] - round[j - 1][0][k][self.x_index]) - (
                                    round[j - 1][0][k][self.x_index] - round[j - 2][0][k][self.x_index])

                        if max_y < round[j][0][k][self.y_index]:
                            max_y = round[j][0][k][self.y_index]
                        if min_y > round[j][0][k][self.y_index]:
                            min_y = round[j][0][k][self.y_index]
                        if max_d_y < round[j][0][k][self.y_index] - round[j - 1][0][k][self.y_index]:
                            max_d_y = round[j][0][k][self.y_index] - round[j - 1][0][k][self.y_index]
                        if min_d_y > round[j][0][k][self.y_index] - round[j - 1][0][k][self.y_index]:
                            min_d_y = round[j][0][k][self.y_index] - round[j - 1][0][k][self.y_index]
                        if max_dd_y < (round[j][0][k][self.y_index] - round[j - 1][0][k][self.y_index]) - (
                                round[j - 1][0][k][self.y_index] - round[j - 2][0][k][self.y_index]):
                            max_dd_y = (round[j][0][k][self.y_index] - round[j - 1][0][k][self.y_index]) - (
                                    round[j - 1][0][k][self.y_index] - round[j - 2][0][k][self.y_index])
                        if min_dd_y > (round[j][0][k][self.y_index] - round[j - 1][0][k][self.y_index]) - (
                                round[j - 1][0][k][self.y_index] - round[j - 2][0][k][self.y_index]):
                            min_dd_y = (round[j][0][k][self.y_index] - round[j - 1][0][k][self.y_index]) - (
                                    round[j - 1][0][k][self.y_index] - round[j - 2][0][k][self.y_index])
                label = [[], [], [], [], []]
                for j in range(i, i + self.observe_len):
                    seq.append([
                        # [self.normalize(round[j][0][k][self.speed_index], max_speed, min_speed) for k in
                        #  range(0, self.num_objects)],
                        # speed
                        [self.normalize((round[j][0][k][self.x_index] - round[j - 1][0][k][self.x_index]) - (
                                round[j - 1][0][k][self.x_index] - round[j - 2][0][k][self.x_index]), max_dd_x,
                                        min_dd_x)
                         for k in
                         range(0, self.num_objects)],
                        [self.normalize(round[j][0][k][self.x_index] - round[j - 1][0][k][self.x_index], max_d_x,
                                        min_d_x)
                         for k in
                         range(0, self.num_objects)],
                        [self.normalize(round[j][0][k][self.x_index], max_x, min_x)
                         for k in
                         range(0, self.num_objects)],
                        [self.normalize((round[j][0][k][self.y_index] - round[j - 1][0][k][self.y_index]) - (
                                round[j - 1][0][k][self.y_index] - round[j - 2][0][k][self.y_index]), max_dd_y,
                                        min_dd_y)
                         for k in
                         range(0, self.num_objects)],

                        [self.normalize(round[j][0][k][self.y_index] - round[j - 1][0][k][self.y_index], max_d_y,
                                        min_d_y)
                         for k in
                         range(0, self.num_objects)],
                        [self.normalize(round[j][0][k][self.y_index], max_y, min_y)
                         for k in
                         range(0, self.num_objects)],

                        # y
                        # [round[j][0][k][self.l_index] + 1 for k in
                        #  range(0, self.num_objects)],
                        # l
                    ])
                    for k in range(0, 5):
                        label[k].append(self.normalize(round[j][0][k][self.speed_index], max_speed, min_speed))

                        # j = i + self.observe_len + self.label_len
                        # label = [
                        #     [self.normalize(round[j][0][k][self.speed_index], max_speed, min_speed) for k in
                        #      range(0, self.num_objects)],
                        #     [self.normalize(round[j][0][k][self.x_index], max_x, min_x) for k in
                        #      range(0, self.num_objects)],
                        #     [self.normalize(round[j][0][k][self.y_index], max_y, min_y) for k in
                        #      range(0, self.num_objects)]
                        # ]

                test = round[i + self.observe_len + self.label_len - 2][1]

                seq = tensor(seq, device=self.device)
                label = tensor(label, device=self.device)

                self.set.append((seq, label, test, max_speed, min_speed))

    def __getitem__(self, index):
        return self.set[self.start + index]

    def __len__(self):
        return min(self.end - self.start, len(self.set))


class DrivingDataMadule(pl.LightningDataModule):
    def __init__(self, version, train_len, validate_len, test_len, observe_len=5, label_len=1):
        super().__init__()
        self.train_dataset_address = os.path.realpath('.') + f'/dataset/{version}/normal.output'
        self.test_dataset_address = os.path.realpath('.') + f'/dataset/{version}/abnormal.output'
        self.train_len = train_len
        self.validate_len = validate_len
        self.test_len = test_len
        self.observe_len = observe_len
        self.label_len = label_len

    def train_dataloader(self):
        return DataLoader(DrivingDataset(address=self.train_dataset_address, start=0, end=self.train_len,
                                         observe_len=self.observe_len, label_len=self.label_len),
                          shuffle=True, num_workers=0, batch_size=1)

    def val_dataloader(self):
        return DataLoader(
            DrivingDataset(address=self.train_dataset_address, start=self.train_len,
                           end=self.train_len + self.validate_len,
                           observe_len=self.observe_len, label_len=self.label_len),
            shuffle=False, num_workers=0, batch_size=1)

    def test_dataloader(self):
        return DataLoader(
            DrivingDataset(address=self.test_dataset_address, start=0, end=self.test_len,
                           observe_len=self.observe_len, label_len=self.label_len),
            shuffle=False, num_workers=0, batch_size=1)
