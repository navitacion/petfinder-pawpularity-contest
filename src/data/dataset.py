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
        self.img_path = [str(p) for p in img_path.glob(f'**/*.jpg')]


    def __len__(self):
        return len(self.df)

    def _resize_same_aspect(self, img, min_size=380):
        height, width, _ = img.shape

        if min_size < min(height, width):
            aspect_ratio = max(width / height, height / width)

            if height < width:
                rep_size = (round(min_size * aspect_ratio), min_size)  # (w, h)
            else:
                rep_size = (min_size, round(min_size * aspect_ratio))  # (w, h)

            img = cv2.resize(img, rep_size, interpolation=cv2.INTER_AREA)

            return img

        else:
            return img


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target_img_id = row['Id']

        # Image  ----------------------------------------
        target_img_path = [p for p in self.img_path if target_img_id in p]
        img = cv2.imread(target_img_path[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Aspect固定で事前リサイズ
        if self.cfg.data.pre_resize > 0 and self.phase == 'train':
            img = self._resize_same_aspect(img, min_size=self.cfg.data.pre_resize)

        # Validの場合はマストで事前リサイズする
        if self.phase != 'train':
            img = self._resize_same_aspect(img, min_size=int(self.cfg.data.img_size * 1.5))

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