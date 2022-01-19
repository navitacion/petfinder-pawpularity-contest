import pandas as pd
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

        if self.cfg.model.type == 'classification':
            # Binning
            self.df['Pawpularity'] = pd.cut(self.df['Pawpularity'].values, bins=self.cfg.model.cat_definition, labels=False)
            self.df.dropna(inplace=True)
            self.df['Pawpularity'] = self.df['Pawpularity'].astype(int)

            # Rebalancing
            rebalanced_df = pd.DataFrame()
            min_num_sampling = min(self.df['Pawpularity'].value_counts())
            for b in self.df['Pawpularity'].unique():
                tmp = self.df[self.df['Pawpularity'] == b].reset_index(drop=True)
                try:
                    tmp = tmp.sample(n=int(min_num_sampling * 1.2), random_state=self.cfg.data.seed)
                except:
                    tmp = tmp.sample(n=int(min_num_sampling), random_state=self.cfg.data.seed)
                rebalanced_df = pd.concat([rebalanced_df, tmp], axis=0)

            rebalanced_df = rebalanced_df.sample(frac=1.0).reset_index(drop=True)
            self.df = rebalanced_df

            print(self.df['Pawpularity'].value_counts())


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
        # For Regressor Head
        self.train_2_dataset = PetFinderDataset(train, self.cfg, self.transform, phase='val')


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            pin_memory=False,
            num_workers=self.cfg.train.num_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.train.batch_size,
            pin_memory=False,
            num_workers=self.cfg.train.num_workers,
            shuffle=False,
            drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.train.batch_size,
            pin_memory=False,
            num_workers=self.cfg.train.num_workers,
            shuffle=False,
            drop_last=False
        )

    def regressor_dataloader(self):
        return DataLoader(
            self.train_2_dataset,
            batch_size=self.cfg.train.batch_size,
            pin_memory=False,
            num_workers=self.cfg.train.num_workers,
            shuffle=False,
            drop_last=False
        )