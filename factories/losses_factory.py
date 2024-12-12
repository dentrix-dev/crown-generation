from losses.evalMetrics.chamferDistance import ChamferLoss, DistanceDk, chamferDk
from losses.evalMetrics.HausdorffDistance import HausdorffLoss

# Dictionary that maps model names to model classes
LOSS_FACTORY = {
    ###### Generation Losses
    "hausdorff": HausdorffLoss,
    "chamfer": ChamferLoss,
    "l2": DistanceDk,
    "chamfer_l2": chamferDk,
}

def get_loss(name, **kwargs):
    """Fetch the Loss from the factory."""
    if name not in LOSS_FACTORY:
        raise ValueError(f"Loss {name} is not available.")
    return LOSS_FACTORY[name](**kwargs)
