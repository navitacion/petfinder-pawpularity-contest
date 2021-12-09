import hydra
import os
import time
import shutil
from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
import wandb

from src.model.regressor import LGBMModel, Trainer, SVR_Petfinder
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
        name=f'{cfg.regressor.exp_name}',
        tags=[cfg.regressor.type],
        reinit=True)

    logger.log_hyperparams(dict(cfg.regressor))

    # Log Code
    wandb.run.log_code('.', include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

    # Model  ----------------------------------------------------
    if cfg.regressor.type == 'lgbm':
        model = LGBMModel(dict(cfg.regressor.lgbm))
    elif cfg.regressor.type == 'svr':
        model = SVR_Petfinder(dict(cfg.regressor.svr))
    else:
        model = None

    trainer = Trainer(cfg, model)
    trainer.fit()

    wandb_plot(trainer.oof)

    wandb.finish()
    time.sleep(3)

    # Remove checkpoint folder
    shutil.rmtree(cfg.data.asset_dir)



if __name__ == '__main__':
    main()
