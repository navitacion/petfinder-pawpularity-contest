import os
import itertools
import wandb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import pytorch_lightning as pl


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))



class PetFinderLightningRegressor(pl.LightningModule):
    def __init__(self, net, cfg, optimizer=None, scheduler=None):
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
        self.criterion = RMSELoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = 1e+4
        self.weight_paths = []
        self.oof_paths = []
        self.oof = None


    def configure_optimizers(self):
        if self.optimizer is None:
            return [], []
        if self.scheduler is None:
            return [self.optimizer], []
        else:
            scheduler = {
                'scheduler': self.scheduler,
                'interval': 'step',   # Scheduler Step Frequency
                'frequency': 1
            }
            return [self.optimizer], [scheduler]

    def forward(self, img, tabular):
        output = self.net(img, tabular)
        return output

    def step(self, batch):
        img, tabular, label, image_id = batch
        label = label.float()

        out = self.forward(img, tabular)
        loss = self.criterion(out, label.view_as(out))

        del img, tabular

        return loss, label, out, image_id

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.step(batch)
        self.log('train/loss', loss, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, label, logits, image_id = self.step(batch)
        self.log('val/loss', loss, on_epoch=True)

        return {'val_loss': loss, 'logits': logits, 'labels': label, 'image_id': image_id}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logits = torch.cat([x['logits'] for x in outputs]).detach().cpu().numpy().reshape((-1))
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy().reshape((-1))
        self.log('val_loss', avg_loss)

        # Post Process
        logits = np.clip(logits, 0, 100)

        if avg_loss.item() < self.best_loss:
            # Save Weights
            filename = '{}-seed_{}_fold_{}_ims_{}_epoch_{}_loss_{:.3f}.pth'.format(
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


class PetFinderLightningClassifier(pl.LightningModule):
    def __init__(self, net, cfg, optimizer=None, scheduler=None):
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
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = 1e+4
        self.weight_paths = []
        self.oof_paths = []
        self.oof = None


    def configure_optimizers(self):
        if self.optimizer is None:
            return [], []
        if self.scheduler is None:
            return [self.optimizer], []
        else:
            scheduler = {
                'scheduler': self.scheduler,
                'interval': 'step',   # Scheduler Step Frequency
                'frequency': 1
            }
            return [self.optimizer], [scheduler]

    def forward(self, img, tabular):
        output = self.net(img, tabular)
        return output

    def step(self, batch):
        img, tabular, label, image_id = batch
        label = label.float()

        # Replace Value  100 -> 1 other 0 binary
        if torch.cuda.is_available():
            _label = torch.where(label >= 100, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())
        else:
            _label = torch.where(label >= 100, torch.tensor(1.), torch.tensor(0.))

        out = self.forward(img, tabular)
        loss = self.criterion(out, _label.view_as(out))

        del img, tabular

        return loss, label, out, image_id

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self.step(batch)
        self.log('train/loss', loss, on_epoch=True)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, label, logits, image_id = self.step(batch)
        self.log('val/loss', loss, on_epoch=True)

        return {'val_loss': loss, 'logits': torch.sigmoid(logits), 'labels': label, 'image_id': image_id}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logits = torch.cat([x['logits'] for x in outputs]).detach().cpu().numpy().reshape((-1))
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy().reshape((-1))
        self.log('val_loss', avg_loss)

        _labels = np.where(labels == 100, 1, 0)
        auc = roc_auc_score(_labels, logits)
        self.log('val_auc', auc)

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