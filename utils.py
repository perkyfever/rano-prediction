import os
import random
import numpy as np
import torch

RANDOM_SEED = 0xBAD5EED

CLASS_TO_RESPONSE = {0: "CR", 1: "PR", 2: "SD", 3: "PD"}
RESPONSE_TO_CLASS = {"CR": 0, "PR": 1, "SD": 2, "PD": 3}


def seed_everything() -> torch.Generator:
    random.seed(RANDOM_SEED)
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    return generator
