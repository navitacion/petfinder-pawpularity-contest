import gc
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold


class CSVDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def _get_fold(self, df):
        df['fold'] = -1
        # df['bin'] = pd.qcut(df['Pawpularity'].values, 5, labels=False)
        df['bin'] = pd.cut(df['Pawpularity'].values, 14, labels=False)

        # StratifiedKFold
        # binning by Pawpularity
        kf = StratifiedKFold(
            n_splits=self.cfg.data.n_splits,
            shuffle=True,
            random_state=self.cfg.data.seed
        )
        for i, (trn_idx, val_idx) in enumerate(kf.split(df, df['bin'].values)):
            df.loc[val_idx, 'fold'] = i

        df['fold'] = df['fold'].astype(np.float16)

        return df


    def _extract_img_size(self, df, phase='train'):

        img_paths = Path(self.cfg.data.data_dir)
        img_paths = [str(p) for p in img_paths.glob(f'{phase}/**/*.jpg')]

        heights = []
        widths = []
        ids = []

        print(f'Extracting Image Info - {phase}')

        for _id in tqdm(df['Id'].values):
            img_path = [p for p in img_paths if _id in p][0]

            img = Image.open(img_path)
            img = np.array(img)

            h, w, _ = img.shape
            heights.append(h)
            widths.append(w)
            ids.append(_id)

        img_info_df = pd.DataFrame({
            'Id': ids,
            'height': heights,
            'width': widths
        })

        # Normalize
        img_info_df['height'] /= 1280
        img_info_df['width'] /= 1280

        df = df.merge(img_info_df, on='Id')


        return df

    def get_data(self):
        data_dir = Path(self.cfg.data.data_dir)

        train = pd.read_csv(data_dir.joinpath('train.csv'))
        test = pd.read_csv(data_dir.joinpath('test.csv'))

        self._get_fold(train)
        train = self._extract_img_size(train, 'train')
        test = self._extract_img_size(test, 'test')

        train['is_train'] = 1
        test['is_train'] = 0

        # ある閾値を超えるかどうかの分類問題として解く
        if self.cfg.data.target_type == 'classification':
            train['Pawpularity'] = train['Pawpularity'].apply(lambda x: 100 if x > self.cfg.data.cls_th else 0)

        df = pd.concat([train, test], axis=0, ignore_index=True)

        del train, test
        gc.collect()

        return df

