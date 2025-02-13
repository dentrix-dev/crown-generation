from train.train_pct import train

# Factory to choose the suitable training Loop use args
TRAIN_FACTORY = {
    "FoldingNet": train,
    "Generation": train,
}

def get_train(model, *args, **kwargs):
    """Fetch the appropriate PointNet model based on the mode."""
    if model not in TRAIN_FACTORY:
        raise ValueError(f"Mode {model} is not available.")
    return TRAIN_FACTORY[model](*args, **kwargs)
