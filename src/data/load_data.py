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

        df = df.merge(img_info_df, on='Id')


        return df

    def get_data(self):
        data_dir = Path(self.cfg.data.data_dir)

        train = pd.read_csv(data_dir.joinpath('train.csv'))
        test = pd.read_csv(data_dir.joinpath('test.csv'))

        self._get_fold(train)


        # When I use augmented image by CycleGAN, I must set same Id as same fold.
        if self.cfg.data.data_dir == './input_cycle_gan':
            tmp = train.copy()
            tmp2 = train.copy()

            tmp['Id'] = tmp['Id'].apply(lambda x: x + '_monet_generated')
            tmp2['Id'] = tmp2['Id'].apply(lambda x: x + '_vangogh_generated')

            train = pd.concat([train, tmp, tmp2], axis=0, ignore_index=True)
            del tmp, tmp2

        # Add Feature - Image Info
        # train = self._extract_img_size(train, 'train')
        # test = self._extract_img_size(test, 'test')

        train['is_train'] = 1
        test['is_train'] = 0

        df = pd.concat([train, test], axis=0, ignore_index=True)

        del train, test
        gc.collect()

        return df

