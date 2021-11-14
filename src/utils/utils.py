import numpy as np
import wandb
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import transformers
from timm.optim import RAdam

from src.utils.sam import SAM


def wandb_plot(oof):
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


class ValueTransformer:
    def __init__(self):
        pass

    def forward(self, v):
        return v / 100

    def backward(self, v):
        return v * 100


def get_optimizer_sceduler(cfg, net, total_step):
    # optimizer
    if cfg.train.optimizer == 'adam':
        optimizer = optim.Adam(
            net.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            eps=1e-8,
        )

    elif cfg.train.optimizer == 'adamw':
        optimizer = optim.AdamW(
            net.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            eps=1e-8,
        )

    elif cfg.train.optimizer == 'sgd':
        optimizer = optim.SGD(
            net.parameters(),
            lr=cfg.train.lr,
            momentum=0.9
        )

    elif cfg.train.optimizer == 'radam':
        optimizer = RAdam(
            net.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay
        )

    elif cfg.train.optimizer == 'sam':
        optimizer = SAM(
            net.parameters(),
            RAdam,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
    else:
        raise ValueError

    # Scheduler
    if cfg.train.scheduler == 'cosine':
        if isinstance(cfg.train.warmup_step, float):
            warmup_step = int(total_step * cfg.train.warmup_step)
        else:
            warmup_step = cfg.train.warmup_step

        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_step,
            num_training_steps=total_step,
            num_cycles=cfg.train.num_cycles
        )
    elif cfg.train.scheduler == 'cosine_annealing':
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.train.epoch,
            eta_min=cfg.train.lr * 0.01
        )
    else:
        raise ValueError

    return optimizer, scheduler