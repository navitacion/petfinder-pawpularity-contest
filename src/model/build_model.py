
from src.model.cnn import PetFinderModel, PetFinderImageModel
from src.model.hybrid_swin_cnn import PawpularityHybridModel

def get_model(cfg, logger=None):

    if cfg.model.type == 'cnn':
        logger.log_hyperparams(dict(cfg.cnn_model))
        if not cfg.train.image_only:
            net = PetFinderModel(**dict(cfg.cnn_model))
        else:
            net = PetFinderImageModel(**dict(cfg.cnn_model))
    elif cfg.model.type == 'hybrid':
        net = PawpularityHybridModel(**dict(cfg.hybrid_model))
        logger.log_hyperparams(dict(cfg.hybrid_model))
    else:
        net = None

    return net