import gc
import os
import itertools
import pickle

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
import pytorch_lightning as pl

from src.utils.utils import get_optimizer_sceduler, ValueTransformer, get_optimizer_sceduler_sam
from src.system.mixup import mixup, MixupCriterion
from src.system.cutmix import cutmix, CutMixCriterion, resizemix


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))



class PetFinderLightningClassifier(pl.LightningModule):
    def __init__(self, net, cfg, dm):
        """
        ------------------------------------
        Parameters
        net: torch.nn.Module
            Model
        cfg: DictConfig
            Config
        optimizer: torch.optim
            Optimizer
        scheduler: torch.optim.lr_scheduler
            Learning Rate Scheduler
        """
        super(PetFinderLightningClassifier, self).__init__()
        self.net = net
        self.cfg = cfg
        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = SmoothBCEwLogits(smoothing=0.05)
        self.best_loss = 1e+4
        self.best_clf_rmse = 1e+4
        self.weight_paths = []
        self.oof_paths = []
        self.oof = None
        self.feat_map_paths = []
        self.clf_paths = []
        self.value_transformer = ValueTransformer()
        self.automatic_optimization = False if self.cfg.train.use_sam else True
        self.clf = None
        self.dm = dm


    def configure_optimizers(self):

        if self.cfg.train.use_sam:
            optimizer = get_optimizer_sceduler_sam(self.cfg, self.net, self.cfg.data.total_step)
            return [optimizer], []

        else:
            optimizer, scheduler = get_optimizer_sceduler(self.cfg, self.net, self.cfg.data.total_step)
            return [optimizer], [scheduler]


    def _train_regressor(self):
        train_img_feats = []
        train_labels = []
        for img, tabular, label, image_id in self.dm.regressor_dataloader():
            with torch.no_grad():
                img = img.cuda()
                tabular = tabular.cuda()

                _, _feat = self.forward(img, tabular)
                _feat = torch.cat([_feat, tabular], dim=1)
                train_img_feats.append(_feat)
                train_labels.append(label)

        train_img_feats = torch.cat(train_img_feats, dim=0).cpu().numpy()
        train_labels = torch.cat(train_labels, dim=0).cpu().numpy()

        self.clf = SVR(**dict(self.cfg.regressor.svr))
        self.clf.fit(train_img_feats, train_labels)


    def forward(self, img, tabular):
        output, feat_map = self.net(img, tabular)
        # output = self.net(img)
        return output, feat_map

    def step(self, batch, phase='train', rand=0):
        img, tabular, label, image_id = batch
        label = label.float()
        # Labelを変換
        label = self.value_transformer.forward(label)

        lim_epoch = int(self.cfg.train.epoch * 0.5)

        # mixup
        if rand > (1.0 - self.cfg.train.mixup_pct) and phase == 'train' and self.current_epoch < self.cfg.train.epoch - lim_epoch:
            img, tabular, label = mixup(img, tabular, label, alpha=self.cfg.train.mixup_alpha)
            out, feat_map = self.forward(img, tabular)
            loss_fn = MixupCriterion(criterion_base=self.criterion)
            loss = loss_fn(out, label)

        # cutmix
        elif rand > (1.0 - self.cfg.train.cutmix_pct) and phase == 'train' and self.current_epoch < self.cfg.train.epoch - lim_epoch:
            img, tabular, label = cutmix(img, tabular, label, alpha=self.cfg.train.cutmix_alpha)
            out, feat_map = self.forward(img, tabular)
            loss_fn = CutMixCriterion(criterion_base=self.criterion)
            loss = loss_fn(out, label)

        # resizemix
        elif rand > (1.0 - self.cfg.train.resizemix_pct) and phase == 'train' and self.current_epoch < self.cfg.train.epoch - lim_epoch:
            img, tabular, label = resizemix(img, tabular, label, alpha=self.cfg.train.resizemix_alpha)
            out, feat_map = self.forward(img, tabular)
            loss_fn = CutMixCriterion(criterion_base=self.criterion)
            loss = loss_fn(out, label)


        else:
            out, feat_map = self.forward(img, tabular)
            loss = self.criterion(out, label.view_as(out))

        del img

        return loss, label, out, image_id, feat_map, tabular

    def training_step(self, batch, batch_idx):
        rand = np.random.rand()
        # SAM Optimizer
        if self.cfg.train.use_sam:
            opt = self.optimizers()
            loss_1 = self.step(batch, phase='train', rand=rand)
            self.manual_backward(loss_1[0])
            opt.first_step(zero_grad=True)

            loss_2 = self.step(batch, phase='train', rand=rand)
            self.manual_backward(loss_2[0])
            opt.second_step(zero_grad=True)

            self.log('train/loss', loss_1[0], on_epoch=True)

            return loss_1

        # Normal Optimizer
        else:
            loss, label, _, _, feat_map, tabular = self.step(batch, phase='train', rand=rand)
            self.log('train/loss', loss, on_epoch=True)

            if isinstance(label, tuple):
                # new_label = lam * y_a + (1 - lam) * y_b
                label = label[2] * label[0] + (1 - label[2]) * label[1]

            return {'loss': loss, 'labels': label.detach(), 'feat_map': feat_map.detach(), 'tabular': tabular.detach()}


    def validation_step(self, batch, batch_idx):
        loss, label, logits, image_id, feat_map, tabular = self.step(batch, phase='val', rand=0)
        self.log('val/loss', loss, on_epoch=True)

        output = {
            'val_loss': loss,
            'logits': torch.sigmoid(logits).detach(),
            'labels': label.detach(),
            'image_id': image_id,
            'feat_map': feat_map.detach(),
            'tabular': tabular.detach()
        }

        return output


    def validation_epoch_end(self, outputs):
        logits = torch.cat([x['logits'] for x in outputs]).cpu().numpy().reshape((-1))
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy().reshape((-1))
        feat_map = torch.cat([x['feat_map'] for x in outputs]).cpu().numpy()
        tabular = torch.cat([x['tabular'] for x in outputs]).cpu().numpy()
        feat_map = np.concatenate([feat_map, tabular], axis=1)

        # Post Process
        # 予測結果を逆変換
        logits = self.value_transformer.backward(logits)
        labels = self.value_transformer.backward(labels)
        logits = np.clip(logits, 0, 100)
        rmse = np.sqrt(mean_squared_error(labels, logits))

        self.log('val_rmse', rmse)

        # Predict CLF
        self._train_regressor()

        pred_clf = self.clf.predict(feat_map)
        clf_rmse = np.sqrt(mean_squared_error(y_true=labels, y_pred=pred_clf))
        self.log('CLF RMSE', clf_rmse)

        # Logging
        if rmse < self.best_loss:
            filename = '{}-seed_{}_fold_{}_ims_{}_epoch_{}_rmse_{:.3f}.pth'.format(
                self.cfg.train.exp_name, self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, rmse
            )
            filename = os.path.join(self.cfg.data.asset_dir, filename)

            torch.save(self.net.state_dict(), filename)
            self.weight_paths.append(filename)

            self.best_loss = rmse

            # Save oof
            ids = [x['image_id'] for x in outputs]
            ids = [list(x) for x in ids]
            ids = list(itertools.chain.from_iterable(ids))
            self.oof = pd.DataFrame({
                'Id': ids,
                'GroundTruth': labels,
                'Pred': logits
            })

            # Regressor
            filename = 'svr-{}_seed_{}_fold_{}_ims_{}_epoch_{}_rmse_{:.3f}.pkl'.format(
                self.cfg.train.exp_name, self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, clf_rmse
            )
            filename = os.path.join(self.cfg.data.asset_dir, filename)

            with open(filename, 'wb') as f:
                pickle.dump(self.clf, f)
            self.clf_paths.append(filename)

            self.best_clf_rmse = rmse


        return None


    def test_step(self, batch, batch_idx):
        img, tabular, _, image_id = batch
        pred = self.forward(img, tabular)

        return {'preds': pred, 'id': image_id}


    def test_epoch_end(self, outputs) -> None:
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        ids = [x['id'] for x in outputs]
        ids = [list(x) for x in ids]
        ids = list(itertools.chain.from_iterable(ids))

        self.sub = pd.DataFrame({
            'Id': ids,
            'Pawpularity': preds.reshape((-1)).tolist()
        })

        return None
