import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from torchvision.transforms import InterpolationMode

class INaturalistDataLoader(pl.LightningDataModule):
    """
    DataModule for the iNaturalist dataset.
    Loads data and creates stratified train/validation split.
    """
    def __init__(self, data_dir, batch_size=32, num_workers=4, img_size=64, val_split=0.2, use_aug=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.val_split = val_split
        self.use_aug = use_aug

        self.base_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])

        if self.use_aug:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ])
        else:
            self.transform = self.base_transform



    def setup(self, stage=None):
        # Load the train dataset
        train_dir = self.data_dir / "train"
        self.full_train = datasets.ImageFolder(train_dir, transform=self.transform)
        
        # Perform stratified split on targets
        targets = np.array(self.full_train.targets)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_split, random_state=43)
        indices = list(range(len(self.full_train)))
        for train_idx, val_idx in sss.split(indices, targets):
            self.train_idx = train_idx
            self.val_idx = val_idx
        
        self.train_dataset = Subset(self.full_train, self.train_idx)
        self.val_dataset = Subset(self.full_train, self.val_idx)
        
        # Load the test dataset 
        test_dir = self.data_dir / "val"
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

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    data_module = INaturalistDataLoader(data_dir=data_dir, batch_size=8, img_size=64)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape, "Labels:", labels)
