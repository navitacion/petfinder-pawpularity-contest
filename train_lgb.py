import hydra
import os
import time
import shutil
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
import wandb

from src.system.lgbm import LGBMModel, Trainer
from src.utils.utils import wandb_plot

@hydra.main(config_name='config.yaml')
def main(cfg):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    seed_everything(cfg.data.seed)

    try:
        # Remove checkpoint folder
        shutil.rmtree(cfg.data.asset_dir)
    except:
        pass

    os.makedirs(cfg.data.asset_dir, exist_ok=True)

    # Logger  --------------------------------------------------
    load_dotenv('.env')
    wandb.login(key=os.environ['WANDB_KEY'])
    logger = WandbLogger(
        project='PetFinder-Pawpularity-Contest',
        name=f'lgbm-{cfg.train.exp_name}-fold{cfg.train.fold}',
        reinit=True)

    logger.log_hyperparams(dict(cfg.lgbm))

    # Log Code
    wandb.run.log_code('.', include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

    # Model  ----------------------------------------------------
    model = LGBMModel(dict(cfg.lgbm.params))

    trainer = Trainer(cfg, model)
    trainer.fit()

    wandb_plot(trainer.oof)

    wandb.finish()
    time.sleep(3)

    # Remove checkpoint folder
    shutil.rmtree(cfg.data.asset_dir)



if __name__ == '__main__':
    main()
