# PetFinder.my - Pawpularity Contest


Kaggle Competition Repogitory

https://www.kaggle.com/c/petfinder-pawpularity-score

## References

- [Boost CV LB with RAPIDS SVR](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/276724)

## Result

- Private Score: 17.07790
- Rank: 220th / 3545 (7%)


## Getting Started

Easy to do, only type command.

```commandline
docker-compose up --build -d
docker exec -it pet_env bash
```

## Solution

### Model Arch
- Multi Modal (Image, Tabular) with LightGBM
#### CNN
- Backbone: Swin Transformer 224x224, 384x384 with Custom Header
- Image size: 224, 384
- Learning rate: 0.00001
- Batch size: 8, 20
- Epochs: 20
- N_Fold: 7
#### Tabular
- DNN with residual block
- Same Params with CNN
#### LightGBM
- Using as Head layer
  - Input: CNN Feature

### Cross Validation
- StratifiedKFold 
  - 14 bins of Target "Pawpularity"

### Preprocess
- Image Resize(equal with aspect ratio) Only Inference
  - Before scoring all image are transformed resize (image size * 1.5)

### Data Augmentation
- Flip
  - Horizontal
- ShiftScaleRotate
- ColorJitter
- MotionBlur
- CoarseDropout
- Mixup

### Optimization
  - NN
    - Transform 0-1 values with Target "Pawpularity"
    - Minimize BCEWithLogitLoss, not RMSE
  - LightGBM
    - Minimize RMSE

### Don't Work
- CycleGAN Image Generation
- Cutmix, Resizemix
- Classification of Pawpularity
  - ex.
    - binary: Pawpularity = 100 or not
    - multi label: Pawpularity < 20, 20 <= Pawpularity < 60, Pawpularity >= 60
- Efficientnet, ViT


## Model Training

To train Regression Model, execute the following command.
```commandline
python train.py train.exp_name=experiment01
```

To train Classification Model,  execute the following command.
```commandline
python train_cls.py train.exp_name=experiment01
```


## Helper Function

- train.py
  - For training Regression Model

- train_cls.py (not improve LB...)
  - For training Classification Model
  - Set Parameter `cat_definition` defined labels
    - ex. cat_definition = [0, 20, 100] -> 2 label Classification
      - Meanings:
        - label = 0: (0 < 'Pawpularity' < 20)
        - label = 1: (20 < 'Pawpularity' < 100)
        - 