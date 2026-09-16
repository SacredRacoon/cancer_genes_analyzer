import numpy as np
import random
import logging
import os

logger = logging.getLogger(__name__)

class RandomStateManager:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._apply_seed()
        logger.info(f"Random state mangafer init, seed {seed}")

    def _apply_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass

    def get_numpy_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def set_seed(self, seed: int):
        self.seed = seed
        self._apply_seed()
        logger.info(f"Random seed changed {seed}")

        