import torch
import random
import numpy as np

def set_seed(config):
    """Set seed for random number generators in pytorch, numpy and python.random."""
    big_seed = int(config["seed"]) if config.get("seed") is not None else np.random.randint(0, 10**8)
    torch.manual_seed(big_seed)
    torch.cuda.manual_seed_all(big_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(big_seed)
    random.seed(big_seed)
