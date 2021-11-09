import os.path

import wandb

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
    def __init__(self, net, cfg, optimizer, scheduler=None):
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

    def configure_optimizers(self):
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
        # logits = torch.cat([x['logits'] for x in outputs]).reshape((-1))
        # labels = torch.cat([x['labels'] for x in outputs]).reshape((-1))

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

        del avg_loss

        return None