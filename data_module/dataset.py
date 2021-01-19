import os
from torch.utils.data import Dataset
from data_module.XML_parrser import XMLParser
from torch import tensor, device
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class DrivingDataset(Dataset):

    def __init__(self, config):
        self.parser = XMLParser(config['address'])
        self.start = config['start']
        self.end = config['end']
        self.n = config['n']
        self.m = config['m']
        self.k = config['k']
        self.set = []
        self.load_set()

    # def load_set(self):
    #     cuda0 = device('cuda')
    #     raw_data = self.parser.read_xml(self.start, self.end)
    #     for vehicle in raw_data:
    #         for i in range(0, len(raw_data[vehicle]) - self.n - self.m - self.k):
    #             seq = []
    #             label = []
    #             sample = []
    #             for j in range(i, i + self.n):
    #                 seq.append([item[0:7] for item in raw_data[vehicle][j]])
    #             for j in range(i + self.n, i + self.n + self.m):
    #                 label.append([item[0:7] for item in raw_data[vehicle][j]])
    #             self.set.append((tensor(seq, device=cuda0), tensor(label, device=cuda0),
    #                              float(raw_data[vehicle][i + self.n + self.m + self.k - 1][0][7])))

    def load_set(self):
        cuda0 = device('cuda')
        raw_data = self.parser.read_txt()
        for round in raw_data:
            for i in range(0, len(round) - self.n):
                seq = []
                for j in range(i, i + self.n):
                    seq.append(round[j])
                self.set.append(tensor(seq, device=cuda0))

    def __getitem__(self, index):
        return self.set[self.start + index]

    def __len__(self):
        return self.end - self.start


class DrivingDataMadule(pl.LightningDataModule):
    def __init__(self, version, train_len, validate_len, test_len):
        super().__init__()
        # self.train_dataset_address = os.path.realpath('.') + f'/dataset/{version}/normal.output'
        self.train_dataset_address = '/home/sepehr/PycharmProjects/DAD/simulation/simulation.out'
        self.test_dataset_address = '/home/sepehr/PycharmProjects/DAD/simulation/simulation2.out'
        self.train_len = train_len
        self.validate_len = validate_len
        self.test_len = test_len

    def train_dataloader(self):
        return DataLoader(DrivingDataset(
            config={'address': self.train_dataset_address, 'start': 0, 'end': self.train_len, 'n': 5, 'm': 3,
                    'k': 1}),
            shuffle=True, num_workers=0, batch_size=1)

    def val_dataloader(self):
        return DataLoader(DrivingDataset(
            config={'address': self.train_dataset_address, 'start': 1000,
                    'end': 1000 + self.validate_len, 'n': 5,
                    'm': 3,
                    'k': 1}), shuffle=False, num_workers=0, batch_size=1)

    def test_dataloader(self):
        return DrivingDataset(
            config={'address': self.test_dataset_address, 'start': 0,
                    'end': self.test_len, 'n': 5, 'm': 3,
                    'k': 1})
