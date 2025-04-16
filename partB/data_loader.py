import os
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

class INaturalistDataLoader(pl.LightningDataModule):
    """
    DataModule for the iNaturalist dataset.
    Loads data from the folder structure with 'train' and 'val' directories.
    Performs a stratified split on the train folder to create training and validation subsets.
    """
    def __init__(self, data_dir, batch_size=32, num_workers=4, img_size=224, val_split=0.2, use_aug=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.val_split = val_split
        self.use_aug = use_aug

        # Base transform: resize and normalize (using ImageNet stats for pre-trained models)
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ])

        # Augmentation transform
        if self.use_aug:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = self.base_transform

    def setup(self, stage=None):
        # Load the train dataset from the "train" folder.
        train_dir = os.path.join(self.data_dir, "train")
        self.full_train = datasets.ImageFolder(train_dir, transform=self.transform)

        # Perform stratified split on targets to get train and validation subsets.
        targets = np.array(self.full_train.targets)
        indices = list(range(len(self.full_train)))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_split, random_state=43)
        for train_idx, val_idx in sss.split(indices, targets):
            self.train_idx = train_idx
            self.val_idx = val_idx

        self.train_dataset = Subset(self.full_train, self.train_idx)
        self.val_dataset = Subset(self.full_train, self.val_idx)

        # Load the test dataset from the "val" folder.
        test_dir = os.path.join(self.data_dir, "val")
        self.test_dataset = datasets.ImageFolder(test_dir, transform=self.base_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

