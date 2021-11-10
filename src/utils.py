import numpy as np
import wandb


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