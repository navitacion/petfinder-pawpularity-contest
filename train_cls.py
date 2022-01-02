import gc

import hydra
import os
import time
import shutil
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
import wandb

from src.system.dm import PetFinderDataModule
from src.system.lightning import PetFinderLightningRegressor, PetFinderLightningClassifier
from src.model.build_model import get_model
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

    # Configをハードコーディング
    cfg.model.type = 'classification'
    cfg.train.img_mix_pct = 0.0

    assert len(cfg.model.cat_definition) - cfg.cnn_model.out_dim == 1, 'You must check out_dim, cat_definition values'


    # Logger  --------------------------------------------------
    load_dotenv('.env')
    wandb.login(key=os.environ['WANDB_KEY'])
    logger = WandbLogger(
        project='PetFinder-Pawpularity-Contest',
        name=f'cls-{cfg.train.exp_name}-fold{cfg.train.fold}',
        tags=['cls'],
        reinit=True)

    logger.log_hyperparams(dict(cfg.data))
    logger.log_hyperparams(dict(cfg.train))
    logger.log_hyperparams(dict(cfg.aug_kwargs))
    logger.log_hyperparams(dict(cfg.model))

    # Log Code
    wandb.run.log_code('.', include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))

    # Data Module  -----------------------------------------------------
    dm = PetFinderDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    # Get total step for scheduler
    total_step = (len(dm.trainval) // cfg.train.batch_size) * cfg.train.epoch
    cfg.data.total_step = total_step

    # Model  -----------------------------------------------------
    net = get_model(cfg, logger)

    # Lightning System  -----------------------------------------------------
    model = PetFinderLightningClassifier(net, cfg, dm=dm)

    # Callback  -------------------------------------------------------------
    es = EarlyStopping(monitor='CNN LOSS', mode='min', patience=7)

    # Trainer  ------------------------------------------------
    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.epoch,
        gpus=[0],
        num_sanity_val_steps=0,
        deterministic=True,
        callbacks=[es],
        amp_backend='apex',
        amp_level='O1',
        # fast_dev_run=True,
    )

    # Train
    trainer.fit(model, datamodule=dm)

    # Logging
    # save_top_kで指定した精度が高いweightとoofをwandbに保存する
    for i, (weight, oof) in enumerate(
            zip(reversed(model.weight_paths),
                reversed(model.oof_paths))):

        wandb.save(weight)
        wandb.save(oof)

        if i + 1 == cfg.data.save_top_k:
            break

    wandb_plot(model.oof, cfg, name='cnn')


    # Inference
    # trainer.test(model, datamodule=dm)
    # model.sub.to_csv('submission.csv', index=False)

    wandb.finish()
    time.sleep(3)

    # Remove checkpoint folder
    shutil.rmtree(cfg.data.asset_dir)
    del net, trainer, model, dm
    gc.collect()


if __name__ == '__main__':
    main()
