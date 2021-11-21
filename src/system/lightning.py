import os
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
import torch
from torch import nn
import pytorch_lightning as pl

from src.utils.utils import get_optimizer_sceduler, ValueTransformer
from src.system.mixup import mixup, MixupCriterion
from src.utils.losses import SmoothBCEwLogits


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))



class PetFinderLightningRegressor(pl.LightningModule):
    def __init__(self, net, cfg):
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
        self.weight_paths = []
        self.oof_paths = []
        self.oof = None
        self.value_transformer = ValueTransformer()
        self.automatic_optimization = False if cfg.train.optimizer == 'sam' else True


    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer_sceduler(self.cfg, self.net, self.cfg.data.total_step)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',   # Scheduler Step Frequency
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def forward(self, img, tabular):
        output = self.net(img, tabular)
        # output = self.net(img)
        return output

    def step(self, batch, phase='train', rand=0):
        img, tabular, label, image_id = batch
        label = label.float()
        # Labelを変換
        label = self.value_transformer.forward(label)

        # mixup
        if rand > (1.0 - self.cfg.train.mixup_pct) and phase == 'train' and self.current_epoch < self.cfg.train.epoch - 3:
            img, tabular, label = mixup(img, tabular, label, alpha=self.cfg.train.mixup_alpha)
            out = self.forward(img, tabular)
            loss_fn = MixupCriterion(criterion_base=self.criterion)
            loss = loss_fn(out, label)

        else:
            out = self.forward(img, tabular)
            loss = self.criterion(out, label.view_as(out))

        del img, tabular

        return loss, label, out, image_id

    def training_step(self, batch, batch_idx):
        rand = np.random.rand()
        # SAM Optimizer
        if self.cfg.train.optimizer == 'sam':
            opt = self.optimizers()
            loss_1, _, _, _ = self.step(batch, phase='train', rand=rand)
            self.manual_backward(loss_1)
            opt.first_step(zero_grad=True)

            loss_2, _, _, _ = self.step(batch, phase='train', rand=rand)
            self.manual_backward(loss_2)
            opt.second_step(zero_grad=True)

            loss = (loss_1 + loss_2) / 2

        # Normal Optimizer
        else:
            loss, _, _, _ = self.step(batch, phase='train', rand=rand)
        self.log('train/loss', loss, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, label, logits, image_id = self.step(batch, phase='val', rand=0)
        self.log('val/loss', loss, on_epoch=True)

        return {'val_loss': loss, 'logits': torch.sigmoid(logits), 'labels': label, 'image_id': image_id}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logits = torch.cat([x['logits'] for x in outputs]).detach().cpu().numpy().reshape((-1))
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy().reshape((-1))

        # AUC
        try:
            auc = roc_auc_score(y_true=labels, y_score=logits)
            self.log('val_auc', auc)
        except:
            pass

        # Post Process
        # 予測結果を逆変換
        logits = self.value_transformer.backward(logits)
        labels = self.value_transformer.backward(labels)
        logits = np.clip(logits, 0, 100)
        rmse = np.sqrt(mean_squared_error(labels, logits))

        self.log('val_rmse', rmse)

        if rmse < self.best_loss:
            # Save Weights
            if self.cfg.model.type == 'cnn':
                backbone = self.cfg.cnn_model.backbone
            elif self.cfg.model.type == 'cnn':
                backbone = self.cfg.hybrid_model.backbone
            else:
                backbone = ''

            filename = '{}-seed_{}_fold_{}_ims_{}_epoch_{}_loss_{:.3f}.pth'.format(
                backbone, self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, avg_loss.item()
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

            filename = 'oof-seed_{}_fold_{}_ims_{}_epoch_{}_loss_{:.3f}.csv'.format(
                self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, avg_loss.item()
            )
            filename = os.path.join(self.cfg.data.asset_dir, filename)
            self.oof.to_csv(filename, index=False)
            self.oof_paths.append(filename)

        del avg_loss

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
    def __init__(self, net, cfg):
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
        self.best_loss = 1e+4
        self.weight_paths = []
        self.oof_paths = []
        self.oof = None


    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer_sceduler(self.cfg, self.net, self.cfg.data.total_step)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',   # Scheduler Step Frequency
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def forward(self, img, tabular):
        output = self.net(img, tabular)
        return output


    def step(self, batch, phase='train', rand=0):
        img, tabular, label, image_id = batch
        label = label.long()

        # mixup
        if rand > (1.0 - self.cfg.train.mixup_pct) and phase == 'train' and self.current_epoch < self.cfg.train.epoch - 3:
            img, tabular, label = mixup(img, tabular, label, alpha=self.cfg.train.mixup_alpha)
            out = self.forward(img, tabular)
            loss_fn = MixupCriterion(criterion_base=self.criterion)
            loss = loss_fn(out, label)

        else:
            out = self.forward(img, tabular)
            loss = self.criterion(out, label)

        del img, tabular

        return loss, label, out, image_id

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.step(batch)
        self.log('train/loss', loss, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, label, logits, image_id = self.step(batch)
        self.log('val/loss', loss, on_epoch=True)
        logits = torch.softmax(logits, dim=-1)
        _, logits = torch.max(logits, 1)

        return {'val_loss': loss, 'logits': logits, 'labels': label, 'image_id': image_id}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logits = torch.cat([x['logits'] for x in outputs]).detach().cpu().numpy().reshape((-1))
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy().reshape((-1))
        self.log('val_loss', avg_loss)

        if avg_loss.item() < self.best_loss:
            # Save Weights
            filename = 'classification_{}-seed_{}_fold_{}_ims_{}_epoch_{}_loss_{:.3f}.pth'.format(
                self.cfg.model.backbone, self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, avg_loss.item()
            )
            filename = os.path.join(self.cfg.data.asset_dir, filename)

            torch.save(self.net.state_dict(), filename)
            self.weight_paths.append(filename)

            self.best_loss = avg_loss.item()

            # Save oof
            ids = [x['image_id'] for x in outputs]
            ids = [list(x) for x in ids]
            ids = list(itertools.chain.from_iterable(ids))
            self.oof = pd.DataFrame({
                'Id': ids,
                'GroundTruth': labels,
                'Pred': logits
            })

            filename = 'oof-seed_{}_fold_{}_ims_{}_epoch_{}_loss_{:.3f}.csv'.format(
                self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, avg_loss.item()
            )
            filename = os.path.join(self.cfg.data.asset_dir, filename)
            self.oof.to_csv(filename, index=False)
            self.oof_paths.append(filename)

        del avg_loss

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