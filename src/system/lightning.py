import gc
import os
import itertools
import pickle

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import pytorch_lightning as pl
import lightgbm as lgb

from src.utils.utils import get_optimizer_sceduler, ValueTransformer, get_optimizer_sceduler_sam
from src.system.mixup import mixup, MixupCriterion
from src.system.cutmix import cutmix, CutMixCriterion, resizemix


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))



class PetFinderLightningRegressor(pl.LightningModule):
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
        super(PetFinderLightningRegressor, self).__init__()
        self.net = net
        self.cfg = cfg
        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = SmoothBCEwLogits(smoothing=0.05)
        self.best_loss = 1e+4
        self.best_cnn_rmse = 1e+4
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
            optimizer, scheduler = get_optimizer_sceduler_sam(self.cfg, self.net, self.cfg.data.total_step)
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

        del img, tabular, _feat, label
        gc.collect()
        torch.cuda.empty_cache()

        train_img_feats = torch.cat(train_img_feats, dim=0).cpu().numpy()
        train_labels = torch.cat(train_labels, dim=0).cpu().numpy()

        if self.cfg.regressor.type == 'svr':
            self.clf = SVR(**dict(self.cfg.regressor.svr))
            self.clf.fit(train_img_feats, train_labels)


        elif self.cfg.regressor.type == 'lgbm':
            x_train, x_test, y_train, y_test = train_test_split(train_img_feats, train_labels, test_size=0.1, random_state=0)

            train_data = lgb.Dataset(x_train, label=y_train)
            valid_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

            self.clf = lgb.train(dict(self.cfg.regressor.lgbm),
                                train_data,
                                valid_sets=[valid_data, train_data],
                                valid_names=['eval', 'train'],
                                feature_name='auto',
                                # verbose_eval=5000
                                )

    def _generate_mix_image(self, img, tabular, label):
        rand = np.random.rand()

        img_mix_fn_dict = {
            'mixup': [mixup, MixupCriterion(criterion_base=self.criterion)],
            'cutmix': [cutmix, CutMixCriterion(criterion_base=self.criterion)],
            'resizemix': [resizemix, CutMixCriterion(criterion_base=self.criterion)]
        }

        if rand > (1.0 - self.cfg.train.img_mix_pct):
            img_mix_fn = img_mix_fn_dict[self.cfg.train.img_mix_type][0]
            loss_fn = img_mix_fn_dict[self.cfg.train.img_mix_type][1]
            img, tabular, label = img_mix_fn(img, tabular, label, alpha=self.cfg.train.img_mix_alpha)
            return img, tabular, label, loss_fn

        else:
            return img, tabular, label, None


    def forward(self, img, tabular):
        output, feat_map = self.net(img, tabular)
        # output = self.net(img)
        return output, feat_map

    def step(self, batch, phase='train'):
        img, tabular, label, image_id = batch
        label = label.float()
        # Labelを変換
        label = self.value_transformer.forward(label)

        if phase == 'train':
            img, tabular, label, loss_fn = self._generate_mix_image(img, tabular, label)
            out, feat_map = self.forward(img, tabular)

            if loss_fn is not None:
                loss = loss_fn(out, label)
            else:
                loss = self.criterion(out, label.view_as(out))
        else:
            out, feat_map = self.forward(img, tabular)
            loss = self.criterion(out, label.view_as(out))

        del img

        output = {
            'loss': loss,
            'label': label,
            'pred': torch.sigmoid(out),
            'image_id': image_id,
            'feat_map': feat_map,
            'tabular': tabular
        }

        return output

    def training_step(self, batch, batch_idx):
        # SAM Optimizer
        if self.cfg.train.use_sam:
            opt = self.optimizers()
            opt.zero_grad()
            loss_1 = self.step(batch, phase='train')
            self.manual_backward(loss_1[0])
            opt.first_step(zero_grad=True)

            loss_2 = self.step(batch, phase='train')
            self.manual_backward(loss_2[0])
            opt.second_step(zero_grad=True)

            self.log('train/loss', loss_1[0], on_epoch=True)
            self.log('train/loss2', loss_2[0], on_epoch=True)

            return {'loss': loss_1[0]}

        # Normal Optimizer
        else:
            output = self.step(batch, phase='train')
            self.log('train/loss', output['loss'], on_epoch=True)

            return {'loss': output['loss']}


    def validation_step(self, batch, batch_idx):
        output = self.step(batch, phase='val')
        self.log('val/loss', output['loss'], on_epoch=True)

        output = {
            'val_loss': output['loss'],
            'logit': output['pred'].detach(),
            'label': output['label'].detach(),
            'image_id': output['image_id'],
            'feat_map': output['feat_map'],
            'tabular': output['tabular']
        }

        return output


    def validation_epoch_end(self, outputs):
        logits = torch.cat([x['logit'] for x in outputs]).cpu().numpy().reshape((-1))
        labels = torch.cat([x['label'] for x in outputs]).cpu().numpy().reshape((-1))
        feat_map = torch.cat([x['feat_map'] for x in outputs]).cpu().numpy()
        tabular = torch.cat([x['tabular'] for x in outputs]).cpu().numpy()
        feat_map = np.concatenate([feat_map, tabular], axis=1)

        # Post Process
        # 予測結果を逆変換
        logits = self.value_transformer.backward(logits)
        labels = self.value_transformer.backward(labels)
        logits = np.clip(logits, 0, 100)
        cnn_rmse = np.sqrt(mean_squared_error(labels, logits))

        self.log('CNN RMSE', cnn_rmse)

        # Predict CLF
        self._train_regressor()

        pred_clf = None
        if self.cfg.regressor.type == 'svr':
            pred_clf = self.clf.predict(feat_map)
        elif self.cfg.regressor.type == 'lgbm':
            pred_clf = self.clf.predict(feat_map, num_iteration=self.clf.best_iteration)

        clf_rmse = np.sqrt(mean_squared_error(y_true=labels, y_pred=pred_clf))
        self.log('CLF RMSE', clf_rmse)

        avg_rmse = (cnn_rmse + clf_rmse) / 2
        self.log('AVG RMSE', avg_rmse)

        # Logging
        if cnn_rmse < self.best_loss:
            filename = '{}-seed_{}_fold_{}_ims_{}_epoch_{}_rmse_{:.3f}.pth'.format(
                self.cfg.train.exp_name, self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, cnn_rmse
            )
            filename = os.path.join(self.cfg.data.asset_dir, filename)

            torch.save(self.net.state_dict(), filename)
            self.weight_paths.append(filename)

            self.best_loss = cnn_rmse

            # Save oof
            ids = [x['image_id'] for x in outputs]
            ids = [list(x) for x in ids]
            ids = list(itertools.chain.from_iterable(ids))
            self.oof = pd.DataFrame({
                'Id': ids,
                'GroundTruth': labels,
                'Pred': logits
            })

            self.clf_oof = pd.DataFrame({
                'Id': ids,
                'GroundTruth': labels,
                'Pred': pred_clf
            })

            # Regressor
            filename = '{}-{}_seed_{}_fold_{}_ims_{}_epoch_{}_rmse_{:.3f}.pkl'.format(
                self.cfg.regressor.type, self.cfg.train.exp_name, self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, clf_rmse
            )
            filename = os.path.join(self.cfg.data.asset_dir, filename)

            with open(filename, 'wb') as f:
                pickle.dump(self.clf, f)
            self.clf_paths.append(filename)

            self.best_clf_rmse = clf_rmse
            self.best_cnn_rmse = cnn_rmse

        del self.clf
        gc.collect()

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
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = SmoothBCEwLogits(smoothing=0.05)
        self.best_loss = 1e+4
        self.best_cnn_rmse = 1e+4
        self.best_clf_rmse = 1e+4
        self.weight_paths = []
        self.oof_paths = []
        self.oof = None
        self.feat_map_paths = []
        self.clf_paths = []
        self.clf = None
        self.dm = dm


    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer_sceduler(self.cfg, self.net, self.cfg.data.total_step)
        return [optimizer], [scheduler]


    def forward(self, img, tabular):
        output, feat_map = self.net(img, tabular)
        # output = self.net(img)
        return output, feat_map

    def step(self, batch, phase='train', rand=0):
        img, tabular, label, image_id = batch
        label = label.long()

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
            loss = self.criterion(out, label)

        out = torch.softmax(out, dim=1)
        prob, pred_label = torch.max(out, 1)

        del img

        return loss, label, pred_label, prob, image_id

    def training_step(self, batch, batch_idx):
        rand = np.random.rand()

        # Normal Optimizer
        loss = self.step(batch, phase='train', rand=rand)
        self.log('train/loss', loss[0], on_epoch=True)

        return {'loss': loss[0]}


    def validation_step(self, batch, batch_idx):
        loss, label, pred_labels, probs, image_id = self.step(batch, phase='val', rand=0)
        self.log('val/loss', loss, on_epoch=True)

        output = {
            'val_loss': loss,
            'pred_labels': pred_labels.detach(),
            'labels': label.detach(),
            'probs': probs,
            'image_id': image_id,
        }

        return output


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        pred_labels = torch.cat([x['pred_labels'] for x in outputs]).cpu().numpy().reshape((-1))
        labels = torch.cat([x['labels'] for x in outputs]).cpu().numpy().reshape((-1))
        probs = torch.cat([x['probs'] for x in outputs]).cpu().numpy().reshape((-1))

        accuracy = accuracy_score(labels, pred_labels)
        self.log('CNN LOSS', avg_loss)
        self.log('Accuracy', accuracy)

        # Logging
        if avg_loss < self.best_loss:
            filename = 'cls-{}-seed_{}_fold_{}_ims_{}_epoch_{}_loss_{:.3f}.pth'.format(
                self.cfg.train.exp_name, self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, avg_loss
            )
            filename = os.path.join(self.cfg.data.asset_dir, filename)

            torch.save(self.net.state_dict(), filename)
            self.weight_paths.append(filename)

            self.best_loss = avg_loss

            # Save oof
            ids = [x['image_id'] for x in outputs]
            ids = [list(x) for x in ids]
            ids = list(itertools.chain.from_iterable(ids))
            self.oof = pd.DataFrame({
                'Id': ids,
                'GroundTruth': labels,
                'Pred': pred_labels,
                'Probability': probs
            })

            filename = 'cls_oof-{}-seed_{}_fold_{}_ims_{}_epoch_{}_loss_{:.3f}.csv'.format(
                self.cfg.train.exp_name, self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, avg_loss
            )
            filename = os.path.join(self.cfg.data.asset_dir, filename)
            self.oof.to_csv(filename, index=False)
            self.oof_paths.append(filename)

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
