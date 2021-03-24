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
        b = []
        for r in range(0, len(raw_data) - self.num_objects+1, self.num_objects):
            b.append([tensor(raw_data[k][0], device=self.device) for k in range(r, r + self.num_objects)])
            label = [raw_data[k][2] for k in range(r, r + self.num_objects)]
            self.set.append((b, raw_data[r][1], label))
            b = []

    def __getitem__(self, index):
        return self.set[index]

    def __len__(self):
        return min(self.end - self.start, len(self.set))


class DrivingDataMadule(pl.LightningDataModule):
    def __init__(self, version, train_len, validate_len, test_len, observe_len=5, label_len=1):
        super().__init__()
        self.train_dataset_address = os.path.realpath('.') + f'/dataset/{version}/train.txt'
        self.test_dataset_address = os.path.realpath('.') + f'/dataset/{version}/abnormal.output'
        self.validation_dataset_address = os.path.realpath('.') + f'/dataset/{version}/validate.txt'
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
            DrivingDataset(address=self.validation_dataset_address, start=0,
                           end=self.validate_len,
                           observe_len=self.observe_len, label_len=self.label_len),
            shuffle=False, num_workers=0, batch_size=1)

    def test_dataloader(self):
        return DataLoader(
            DrivingDataset(address=self.test_dataset_address, start=0, end=self.test_len,
                           observe_len=self.observe_len, label_len=self.label_len),
            shuffle=False, num_workers=0, batch_size=1)
