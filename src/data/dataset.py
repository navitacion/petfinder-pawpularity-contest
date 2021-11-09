import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset

from src.constant import TABULAR_FEATURES

class PetFinderDataset(Dataset):
    def __init__(self, df, cfg, transform=None, phase='train'):
        self.df = df
        self.transform = transform
        self.phase = phase
        self.cfg = cfg

        img_path = Path(cfg.data.data_dir)
        data_path = 'test' if self.phase == 'test' else 'train'
        self.img_path = [str(p) for p in img_path.glob(f'{data_path}/**/*.jpg')]


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target_img_id = row['Id']

        # Image  ----------------------------------------
        target_img_path = [p for p in self.img_path if target_img_id in p]
        img = cv2.imread(target_img_path[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img, self.phase)

        # Tabular  --------------------------------------
        tabular_features = row[TABULAR_FEATURES]
        tabular_features = torch.tensor(tabular_features, dtype=torch.float)

        # Label  ----------------------------------------
        if self.phase == 'test':
            label = -1
        else:
            label = row['Pawpularity']
        label = torch.tensor(label, dtype=torch.float)

        return img, tabular_features, label, target_img_id