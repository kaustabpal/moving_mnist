#!/usr/bin/env python3
# @brief:    Pytorch Lightning module for datasets
# @author    Kaustab Pal    [kaustab21@gmail.com]
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from matplotlib import pyplot as plt

class MovingMnistModule(pl.LightningDataModule):
    """A Pytorch Lightning module for Datasets"""

    #def __init__(self, cfg):
    def __init__(self):
        """Method to initizalize the dataset class

        Args:
          cfg: config dict

        Returns:
          None
        """
        super(MovingMnistModule, self).__init__()
        #self.cfg = cfg


    def setup(self, stage=None):
        """Dataloader and iterators for training, validation and test data"""
        ########## Point dataset splits

        train_set = MovingMnistDataset(split="train")
        val_set = MovingMnistDataset(split="val")
        test_set = MovingMnistDataset(split="test")

        ########## Generate dataloaders and iterables

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=2,
            shuffle=True,
            num_workers=0, 
            pin_memory=False,
            drop_last=False,
            timeout=0,
            persistent_workers=False
        )
        #self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            persistent_workers=False
        )
        #self.valid_iter = iter(self.valid_loader)

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            persistent_workers=False
        )

        #self.test_iter = iter(self.test_loader)

        print(
            "Loaded {:d} training, {:d} validation and {:d} test samples.".format(
                len(train_set), len(val_set), (len(test_set))
            )
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader


class MovingMnistDataset(Dataset):
    """Dataset class for a dataset"""

    def __init__(self, split = "Train", cfg=None):
        """Read parameters and data

        Args:
            cfg (dict): Config parameters
            split (str): Data split

        Raises:
            Exception: [description]
        """
        self.cfg = cfg
        self.root_dir = os.environ.get("DATA_RAW")
        self.data = np.load(self.root_dir).transpose(1,0,2,3)
        self.data = torch.from_numpy(self.data).unsqueeze(2)
        if split == "train":
            from_idx = 0
            to_idx = 8000
        elif split == "val":
            from_idx = 8000
            to_idx = 9000
        elif split == "test":
            from_idx = 9000
            to_idx = 10000
        else:
            raise Exception("Split must be train/val/test")

        self.data = self.data[from_idx:to_idx]/255.
        self.dataset_size = self.data.shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        """Generic function to get items from a dataset

        Args:
            idx (int): Sample index

        Returns:
            item: Dataset dictionary item
        """
        data = self.data[idx]
        rand = np.random.randint(10,20)
        input_data = data[:10]
        target_output = data[10:]
        item = {"input": input_data, "target_output": target_output}
        return item



if __name__ == "__main__":
    #config_filename = # location of config file
    #cfg = yaml.safe_load(open(config_filename))
    data = MovingMnistModule()
    data.setup()
    train_loader = data.train_loader
    for i, batch in enumerate(train_loader):
        input_ = batch["input"]
        target_ = batch["target_output"]
        for t in range(input_.shape[1]):
            input_img = input_[0,t,0]*255.
            plt.imshow(input_img)
            plt.show()
        #print(input_.shape)
        #print(target_.shape)
        break


