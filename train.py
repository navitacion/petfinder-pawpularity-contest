import hydra
import os
import time
import shutil
from dotenv import load_dotenv
from torch import optim
from torch.optim import lr_scheduler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from src.system.dm import PetFinderDataModule
from src.system.lightning import PetFinderLightningRegressor
from src.model.cnn import PetFinderModel
from src.utils import wandb_plot

@hydra.main(config_name='config.yaml')
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

    # Data Module  -----------------------------------------------------
    dm = PetFinderDataModule(cfg)

    # Model  -----------------------------------------------------
    net = PetFinderModel(**dict(cfg.model))

    # Optimizer & Scheduler  ------------------------------------------------
    optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epoch, eta_min=0)

    # Lightning System  -----------------------------------------------------
    model = PetFinderLightningRegressor(net, cfg, optimizer, scheduler)

    # Callback  -----------------------------------------------------
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, mode='min')

    # Trainer  ------------------------------------------------
    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.epoch,
        gpus=1,
        num_sanity_val_steps=0,
        callbacks=[early_stopping],
        deterministic=True,
        amp_backend='apex',
        amp_level='O1',
        # fast_dev_run=True,
    )

    # Train
    trainer.fit(model, datamodule=dm)

    # Logging
    wandb.log({'Best RMSE': model.best_loss})
    # save_top_kで指定した精度が高いweightとoofをwandbに保存する
    for weight, oof in zip(model.weight_paths[-cfg.data.save_top_k:], model.oof_paths[-cfg.data.save_top_k:]):
        wandb.save(weight)
        wandb.save(oof)
    wandb_plot(model.oof)


    # Inference
    # trainer.test(model, datamodule=dm)
    # model.sub.to_csv('submission.csv', index=False)

    wandb.finish()
    time.sleep(3)

    # Remove checkpoint folder
    shutil.rmtree(cfg.data.asset_dir)


if __name__ == '__main__':
    main()
