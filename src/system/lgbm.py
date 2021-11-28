import os, gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from abc import ABCMeta, abstractmethod
from sklearn.metrics import mean_squared_error
import wandb
import pickle

from src.data.load_data import CSVDataLoader
from src.constant import TABULAR_FEATURES


# Critetion  ------------------------------------------------------------------------------------------
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Model -----------------------------------------------------------------------------------------------
# Base class
class BaseModel(metaclass=ABCMeta):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        raise NotImplementedError

    def get_feature_importance(self):
        pass


# LightGBM
class LGBMModel(BaseModel):
    def __init__(self, params):
        super(LGBMModel, self).__init__(params)

    def train(self, X_train, y_train, X_val, y_val, feature_name='auto'):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.lgb = lgb.train(self.params,
                               train_data,
                               valid_sets=[valid_data, train_data],
                               valid_names=['eval', 'train'],
                               feature_name=feature_name,
                               verbose_eval=5000,
                               )

        oof = self.lgb.predict(X_val, num_iteration=self.lgb.best_iteration)

        return oof


    def predict(self, X_test):
        pred = self.lgb.predict(X_test, num_iteration=self.lgb.best_iteration)
        return pred

    def get_feature_importance(self):
        return self.lgb.feature_importance()


# Trainer Class ------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.criterion = RMSE


    def _prepare_data(self):
        """
        prepare dataset for training
        ---------------------------------------------
        Parameter
        df: dataframe
            preprocessed data
        mode: str
            If training, 'mode' set 'fit', else 'mode' set 'predict'
        ---------------------------------------------
        Returns
        X_train, y_train, X_test, train_id, test_id, features
        """
        # Load From CSV
        csv_loader = CSVDataLoader(self.cfg)
        self.df = csv_loader.get_data()
        # Use only Train
        self.trainval = self.df[self.df['is_train'] == 1].reset_index(drop=True)

        feature_map = pd.DataFrame()
        feature_map_path = [str(p) for p in Path(self.cfg.lgbm.feature_map_path).glob('*.csv')]
        for path in feature_map_path:
            tmp = pd.read_csv(path)
            feature_map = pd.concat([feature_map, tmp], axis=0, ignore_index=True)


        assert self.trainval.shape[0] == feature_map.shape[0]

        self.trainval = pd.merge(self.trainval, feature_map, on='Id', how='left').reset_index(drop=True)

        # Feature Name
        self.feature_names = [f for f in self.trainval.columns if 'feature' in f]
        self.feature_names = self.feature_names + TABULAR_FEATURES


    def _train_cv(self):
        """
        Train loop for Cross Validation
        """
        # init Model list
        self.models = []
        self.oof_pred = np.zeros(len(self.trainval))
        self.oof_y = np.zeros(len(self.trainval))

        for i in range(self.df['fold'].nunique()):
            print(f'Fold {i} Starting...')

            trn_idx = self.trainval[self.trainval['fold'] != i].index
            val_idx = self.trainval[self.trainval['fold'] == i].index

            X_trn = self.trainval.iloc[trn_idx][self.feature_names].values
            y_trn = self.trainval.iloc[trn_idx]['Pawpularity'].values
            X_val = self.trainval.iloc[val_idx][self.feature_names].values
            y_val = self.trainval.iloc[val_idx]['Pawpularity'].values

            oof = self.model.train(X_trn, y_trn, X_val, y_val, feature_name=self.feature_names)

            # Score
            score = self.criterion(y_val, oof)

            # Logging
            print(f'Fold {i + 1}  Score: {score:.3f}')
            wandb.log({'RMSE': score}, step=i)
            self.oof_pred[val_idx] = oof
            self.oof_y[val_idx] = y_val
            self.models.append(self.model.lgb)


    def _train_end(self):
        """
        End of Train loop per crossvalidation fold
        Logging and oof file
        """
        # Log params
        self.oof_score = self.criterion(self.oof_y, self.oof_pred)
        print(f'All Score: {self.oof_score:.3f}')
        wandb.log({'Best RMSE': self.oof_score})

        self.oof = pd.DataFrame({
            'Id': self.trainval['Id'].values,
            'Pred': self.oof_pred,
            'GroundTruth': self.oof_y
        })

        for i, model in enumerate(self.models):
            filename = 'pretrained_lgbm_{}_fold_{}.pkl'.format(self.cfg.train.exp_name, i)
            filename = os.path.join(self.cfg.data.asset_dir, filename)

            with open(filename, 'wb') as f:
                pickle.dump(model, f)
                wandb.save(filename)
                time.sleep(3)


    def fit(self):
        self._prepare_data()
        self._train_cv()
        self._train_end()
