import os
import itertools
import wandb
import numpy as np
import pandas as pd
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


    def _wandb_plot(self, oof):
        # Table
        table = wandb.Table(dataframe=oof)

        # Histogram
        wandb.log({'histgram': wandb.plot.histogram(table, "Pred", title="Pred")})

        # Scatter
        wandb.log({"Scatter" : wandb.plot.scatter(table, "GroundTruth", "Pred")})

        # Confusion Matrix
        # 10の位までで切り上げることで10段階のクラス分類として表現
        oof['Pred'] = oof['Pred'].apply(lambda x: 0 if x < 0 else x)
        ground_truth_ceil = np.ceil(oof['GroundTruth'].values / 10).astype(int)
        pred_ceil = np.ceil(oof['Pred'].values / 10).astype(int)

        cm = wandb.plot.confusion_matrix(
            y_true=ground_truth_ceil,
            preds=pred_ceil,
            class_names=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        wandb.log({"conf_mat": cm})


    def configure_optimizers(self):
        if self.optimizer is None:
            return [], []
        if self.scheduler is None:
            return [self.optimizer], []
        else:
            return [self.optimizer], [self.scheduler]

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
            wandb.save(filename)

            self.best_loss = avg_loss.item()

            # Save oof
            ids = [x['image_id'] for x in outputs]
            ids = [list(x) for x in ids]
            ids = list(itertools.chain.from_iterable(ids))
            oof = pd.DataFrame({
                'Id': ids,
                'GroundTruth': labels,
                'Pred': logits
            })

            filename = 'oof-seed_{}_fold_{}_ims_{}_epoch_{}_loss_{:.3f}.csv'.format(
                self.cfg.data.seed, self.cfg.train.fold,
                self.cfg.data.img_size, self.current_epoch, avg_loss.item()
            )
            filename = os.path.join(self.cfg.data.asset_dir, filename)
            oof.to_csv(filename, index=False)
            wandb.save(filename)

            self._wandb_plot(oof)


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