import hydra
import os
import shutil
from dotenv import load_dotenv
from torch import optim
from torch.optim import lr_scheduler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.system.dm import PetFinderDataModule
from src.system.lightning import PetFinderLightningRegressor
from src.model.cnn import Timm_model, PetFinderModel

@hydra.main(config_path='.', config_name='config')
def main(cfg):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)

    try:
        # Remove checkpoint folder
        shutil.rmtree(cfg.data.asset_dir)
    except:
        pass

    os.makedirs(cfg.data.asset_dir, exist_ok=True)

    # Logger  --------------------------------------------------
    load_dotenv('.env')
    wandb.login(key=os.environ['WANDB_KEY'])
    logger = WandbLogger(project='PetFinder-Pawpularity-Contest', reinit=True)

    logger.log_hyperparams(dict(cfg.data))
    logger.log_hyperparams(dict(cfg.model))
    logger.log_hyperparams(dict(cfg.train))
    logger.log_hyperparams(dict(cfg.aug_kwargs))

    # Log Code
    wandb.run.log_code('.', include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

    dm = PetFinderDataModule(cfg)

    net = PetFinderModel(**dict(cfg.model))

    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0)

    model = PetFinderLightningRegressor(net, cfg, optimizer, scheduler)

    # Trainer  ------------------------------------------------
    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.epoch,
        gpus=1,
        num_sanity_val_steps=0,
        deterministic=True,
        # fast_dev_run=True,
    )

    # Train
    trainer.fit(model, datamodule=dm)
    wandb.log({'Best RMSE': model.best_loss})


if __name__ == '__main__':
    main()
