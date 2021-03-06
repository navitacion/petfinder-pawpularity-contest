
from src.model.cnn import PetFinderModel
from src.model.hybrid_swin_cnn import PawpularityHybridModel

def get_model(cfg, logger=None):

    if cfg.model.type == 'hybrid':
        net = PawpularityHybridModel(**dict(cfg.hybrid_model))
        logger.log_hyperparams(dict(cfg.hybrid_model))
    else:
        net = PetFinderModel(**dict(cfg.cnn_model))
        logger.log_hyperparams(dict(cfg.cnn_model))

    return net
