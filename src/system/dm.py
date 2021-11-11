from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.data.dataset import PetFinderDataset
from src.data.load_data import CSVDataLoader
from src.data.transform import ImageTransform

def collate_fn(batch):
    return tuple(zip(*batch))


class PetFinderDataModule(pl.LightningDataModule):
    """
    """
    def __init__(self, cfg):
        super(PetFinderDataModule, self).__init__()
        self.cfg = cfg

    def prepare_data(self):
        # Load From CSV
        csv_loader = CSVDataLoader(self.cfg)
        self.df = csv_loader.get_data()

        # Define Augmentation
        self.transform = ImageTransform(self.cfg)


    def setup(self, stage=None):
        # Sprit Train and Test
        self.trainval = self.df[self.df['is_train'] == 1]
        test = self.df[self.df['is_train'] == 0]

        # Split by fold
        train = self.trainval[self.trainval['fold'] != self.cfg.train.fold]
        val = self.trainval[self.trainval['fold'] == self.cfg.train.fold]

        # Dataset
        # Train
        self.train_dataset = PetFinderDataset(train, self.cfg, self.transform, phase='train')
        # Valid
        self.valid_dataset = PetFinderDataset(val, self.cfg, self.transform, phase='val')
        # Test
        self.test_dataset = PetFinderDataset(test, self.cfg, self.transform, phase='test')


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            pin_memory=False,
            num_workers=self.cfg.train.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.train.batch_size,
            pin_memory=False,
            num_workers=self.cfg.train.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.train.batch_size,
            pin_memory=False,
            num_workers=self.cfg.train.num_workers,
            shuffle=False
        )
