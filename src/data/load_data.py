import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class CSVDataLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def _get_fold(self, df):
        df['fold'] = -1
        df['bin'] = pd.qcut(df['Pawpularity'].values, 5, labels=False)
        # df['bin'] = pd.cut(df['Pawpularity'].values, 14, labels=False)

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

    def get_data(self):
        data_dir = Path(self.cfg.data.data_dir)

        train = pd.read_csv(data_dir.joinpath('train.csv'))
        test = pd.read_csv(data_dir.joinpath('test.csv'))

        self._get_fold(train)

        train['is_train'] = 1
        test['is_train'] = 0

        # ある閾値を超えるかどうかの分類問題として解く
        if self.cfg.data.target_type == 'classification':
            train['Pawpularity'] = train['Pawpularity'].apply(lambda x: 100 if x > self.cfg.data.cls_th else 0)

        df = pd.concat([train, test], axis=0, ignore_index=True)

        del train, test
        gc.collect()

        return df

