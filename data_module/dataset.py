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

    def load_set(self):
        cuda0 = device('cuda')
        raw_data = self.parser.read_xml(self.start, self.end)
        for vehicle in raw_data:
            for i in range(0, len(raw_data[vehicle]) - self.n - self.m - self.k):
                seq = []
                label = []
                sample = []
                for j in range(i, i + self.n):
                    seq.append(raw_data[vehicle][j])
                for j in range(i + self.n, i + self.n + self.m):
                    label.append(raw_data[vehicle][j])
                for l in range(i + self.n + self.m, i + self.n + self.m + self.k):
                    sample.append(raw_data[vehicle][l])
                self.set.append((tensor(seq, device=cuda0), tensor(label, device=cuda0), tensor(sample, device=cuda0)))

    def __getitem__(self, index):
        return self.set[index]

    def __len__(self):
        return len(self.set)


class DrivingDataMadule(pl.LightningDataModule):
    def __init__(self, version, train_len, validate_len, test_len):
        super().__init__()
        self.train_dataset_address = os.path.realpath('.') + f'/dataset/{version}/normal.output'
        self.test_dataset_address = os.path.realpath('.') + f'/dataset/{version}/abnormal.output'
        self.train_len = train_len
        self.validate_len = validate_len
        self.test_len = test_len

    def train_dataloader(self):
        return DataLoader(DrivingDataset(
            config={'address': self.train_dataset_address, 'start': 0, 'end': self.train_len, 'n': 5, 'm': 3, 'k': 1}),
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(DrivingDataset(
            config={'address': self.train_dataset_address, 'start': self.train_len,
                    'end': self.train_len + self.validate_len, 'n': 5,
                    'm': 3,
                    'k': 1}), shuffle=False)

    def test_dataloader(self):
        return DrivingDataset(
            config={'address': self.test_dataset_address, 'start': 0,
                    'end': self.test_len, 'n': 5, 'm': 3,
                    'k': 1})
