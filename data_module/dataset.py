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
        raw_data = self.parser.read_txt()
        for round in raw_data:
            for i in range(0, len(round) - self.n - self.m):
                seq = []
                label = 0
                for j in range(i, i + self.n):
                    seq.append(round[j][0])
                label = round[i + self.n + self.m][0]
                seq = tensor(seq, device=cuda0)
                seq /= 20
                # mean, std = seq.mean(), seq.std()
                # seq = seq - mean
                # seq = seq / std

                an = round[i + self.n + self.m][1]
                label = tensor(label, device=cuda0)
                label /= 20
                # mean, std = label.mean(), label.std()
                # label = label - mean
                # label = label / std

                self.set.append((seq, label, an))

    def __getitem__(self, index):
        return self.set[self.start + index]

    def __len__(self):
        return min(self.end - self.start, len(self.set))


class DrivingDataMadule(pl.LightningDataModule):
    def __init__(self, version, train_len, validate_len, test_len):
        super().__init__()
        self.train_dataset_address = os.path.realpath('.') + f'/dataset/{version}/normal.output'
        self.test_dataset_address = os.path.realpath('.') + f'/dataset/{version}/abnormal2.output'
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
