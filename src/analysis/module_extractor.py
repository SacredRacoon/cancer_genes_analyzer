import numpy as np
import logging
from typing import Tuple, List, Dict
from scipy.optimize import linear_sum_assignment
from sklearn.utils import resample

logger = logging.getLogger(__name__)

class ModuleExtractor:
    def __init__(self, config: dict):
        cfg = config.get('nmf',{})

        self.k_range = cfg.get('k_range', [4,10])
        self.max_iter = cfg.get('max_iter', 500)
        self.tol = cfg.get('tol', 1e-4)
        self.random_state = cfg.get('random_state',228)

        self.alpha_l1 = cfg.get('alpha_l1', 0.1)
        self.lamda_anchor = cfg.get('lamds_anchor', 0.5)
        self.anchor_freq-threshold = cfg.get('anchor_freq_threshold',0.10)
        self.anchor_var_threshold = cfg.get('anchor_var_threshold',0.01)

        self.soft_thresholding = cfg.get('soft_thresholding',{
            'mutated': 0.9,
            'not_mutated': 0.1,
            'nan': 0.5
        })

        self.stability_n_iterations = cfg.get('stability_n_iterations', 100)
        self.stability_subsample_ratio = cfg.get('stability_subsample_ratio', 0.8)
        self.stability_pi_threshold = cfg.get('stability_pi_threshold', 0.7)
        self.stability_top_n_genes = cfg.get('stability_top_n_genes', 10)

        self.huber_delta = cfg.get('huber_delta', 1.0)

        logger.info(
            f"ModuleExtractor init k_range {self.k_range}, alpha_l1 {self.alpha_l1}, "
            f"lambda_anchor {self.lamda_anchor}, stability_iters = {self.stability_n_iterations}"
        )

    def extract_modules(self, V: np.ndarray, gene_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        logger.info(f"Starting ASW-NMF-SS matrix shape {V.shape}")

        logger.info("Stage 1 soft thresholding")
        V_soft, M = self._soft_threshold(V)

        logger.info("Stage 2 contrast background substract")
        V_contrastive = self._contrastive_substraction(V_soft, M)

    def _soft_threshold(self, V:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        V_soft = np.full_like(V, self.soft_thresholding['nan'],dtype=float)
        M = np.zeros_like(V, dtype=float)

        nan_mask = np.isnan(V)
        not_nan_mask = ~nan_mask

        V_soft[not_nan_mask & (V==1)] = self.soft_thresholding['mutated']
        V_soft[not_nan_mask & (V==0)] = self.soft_thresholding['not_mutated']
        V_soft[nan_mask] = self.soft_thresholding['nan']

        M[not_nan_mask] = 1.0
        M[nan_mask] = 0.0

        logger.info(f"Soft thresholdin {not_nan_mask.sum()} tested, {nan_mask.sum()} NaN ({nan_mask.mean()*100:.1f`}%)")

        return V_soft, M

    def _contrastive_substraction(self, V: np.ndarray, M: np.ndarray) -> np.ndarray:
        background = np.nanmedian(V, axis = 0)

        V_centered = V - background
        V_contrastive = np.maximum(V_centered, 0)
        V_contrastive[np.isnan(V)] = 0.0

        logger.info(f"Contrast substract background median range {np.nanmin(background):.3f}, {np.nanmax(background):.3f}")

        return V_contrastive