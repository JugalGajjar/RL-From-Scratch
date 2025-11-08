from dataclasses import dataclass
import os, random, numpy as np
import torch

@dataclass
class Seed:
    seed: int = 42
    def set(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        torch.manual_seed(self.seed)
        if torch.backends.mps.is_available():
            print("MPS is available, setting the seed for MPS backend.")
