from models.GraphCNN.DGCNN import DGCNN

from models.FoldingNet.Mining import GaussianKernelConv
from models.FoldingNet.FoldingNet import FoldingNet


# Dictionary that maps model names to model classes
MODEL_FACTORY = {
    "KCNet": GaussianKernelConv,
    "FoldingNet": FoldingNet,
    "DynamicGraphCNN": DGCNN,
    # "PCTGeometryAware":PCTransformer,
}

def get_model(name, **kwargs):
    """Fetch the model from the factory."""
    if name not in MODEL_FACTORY:
        raise ValueError(f"Model {name} is not available.")
    return MODEL_FACTORY[name](**kwargs)
