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

        logger.info("Stage 3 adaptive anchor weights")
        anchor_weights, anchor_vectors = self._compute_anchor_weights(V_soft, M, gene_names)

        logger.info("Stage 4 Stability selection")
        optimal_k, selection_probs, stability_scores = self._stability_selection(
            V_contrastive, M, anchor_weights, anchor_vectors
        )
        logger.info(f"Optimal k {optimal_k}, stability score {stability_scores[optimal_k]:.3f}")
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

    def _compute_anchor_weights(self, V: np.ndarray, M: np.ndarray, gene_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n_genes = V.shape[1]
        anchor_weights = np.ones(n_genes, dtype=float)
        anchor_vectors = None

        for i in range(n_genes):
            gene_col = V[:, i]
            mask = M[:, i] == 1.0

            if mask.sum() == 0:
                continue

            freq = np.nanmean(gene_col[mask])

            var = np.nanvar(gene_col[mask])

            freq_factor = min(freq / self.anchor_freq_threshold, 2.0)
            var_factor = min(var / self.anchor_var_threshold, 1.0)
            anchor_weights[i] = min(1.0 + freq_factor + var_factor, 3.0)

        top_anchors = np.argsort(anchor_weights)[-10][::-1]
        logger.info("Top 10 anchor genes")
        for idx in top_anchors:
            logger.info(f"{gene_names[idx]} weight {anchor_weights[idx]:.2f}")

        return anchor_weights, anchor_vectors

    def _weighted_nmf(self, V: np.ndarray, M: np.ndarray, k: int, anchor_weights: np.ndarray, anchor_vectors: np.ndarray = None, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        if random_state is None:
            random_state = self.random_state

        rng = np.random.default_rng(random_state)
        n_patients, n_genes = V.shape

        W = rng.uniform(0.01, 1.0, size=(n_patients, k))
        H = rng.uniform(0.01, 1.0, size=(k,n_genes))

        if anchor_vectors is None:
            anchor_vectors = np.zeros((n_genes, k), dtype=float)
            strong_anchors = anchor_weights > 2.0
            anchor_vectors[strong_anchors, 0] = 1.0

        eps = 1e-10

        prev_loss = np.inf

        for iteration in range(self.max_iter):
            numerator_H = W.T @ (M*V)

            WH = W@H
            denominator_H = W.T @ (M * WH)
            denominator_H += self.alpha_l1

            anchor_penalty = 2 * self.lamda_anchor * np.diag(anchor_weights) @ (H.T - anchor_vectors)

            H = H * (numerator_H + eps) / (denominator_H + anchor_penalty.T + eps)
            H = np.maximum(H, 0)

            numerator_W = (M*V) @ H.T

            WH = W @ H
            denominator_W = (M * WH) @ H.T

            W = W * (numerator_W + eps) / (denominator_W + eps)
            W = np.maximum(W, 0)

            if iteration % 10 == 0:
                WH = W @ H
                residual = M * (V - WH)

                loss = self._huber_loss(residual)
                loss += self.alpha_l1 * np.sum(np.abs(H))
                loss += self.lamda_anchor * np.sum(anchor_weights[:, None] * (H.T - anchor_vectors) ** 2)

                if abs(prev_loss - loss) / (prev_loss + eps) < self.tol:
                    logger.debug(f"NMF converged at iteration {iteration}, loss {loss:.4f}")
                    break

                prev_loss = loss

        return W,H

    def _huber_loss(self, residual: np.ndarray) -> float:
        delta = self.huber_delta
        abs_r = np.abs(residual)

        quadratic = np.minimum(abs_r, delta) **2 / 2
        linear = delta * (np.maximum(abs_r, delta) - delta)

        return np.sum(quadratic + linear)
    
    def _stability_selection(self, V: np.ndarray, M: np.ndarray, anchor_weights: np.ndarray, anchor_vectors: np.ndarray) -> Tuple[int, np.ndarray, Dict[int, float]]:
        n_patients, n_genes =  V.shape
        n_subsample = int(self.stability_subsample_ratio * n_patients)

        max_k = max(self.k_range)
        selectiont_counts = np.zeros((n_genes, max_k))

        stability_scores = {}

        for k in self.k_range:
            logger.info(f"Testing k {k} with {self.stability_n_iterations} subsampling iterations")

            all_H_runs = []

            for iteration in range(self.stability_n_iterations):
                V_sub, M_sub = self._subsample(self.stability_n_iterations)
                W_sub, H_sub = self._weighted_nmf(V_sub, M_sub, k, anchor_weights, anchor_vectors, random_state=iteration)

                all_H_runs.append(H_sub)

                for module_idx in range(k):
                    top_genes = np.argsort(H_sub[module_idx, :])[-self.stability_top_n_genes:]
                    selectiont_counts[top_genes, module_idx] += 1

                selection_probs_k = selectiont_counts[:, :k] / self.stability_n_iterations

                stability = self._compute_stability(all_H_runs, k)
                stability_scores[k] = stability

                logger.info(f"k {k}, stbility {stability:.3f}")

            optimal_k = max(stability_scores, key=stability_scores.get)
            final_selection_probs = selectiont_counts[:,:optimal_k] / self.stability_n_iterations

            return optimal_k, final_selection_probs, stability_scores


    def _subsample(self, V: np.ndarray, M: np.ndarray, n_samples: int, random_state: int) -> Tuple[np.ndarray. np.ndarray]:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(V.shape[0], size=n_samples, replace=False)
        return V[indices,:], M[indices, :]

    def _compute_stability(self, H_runs: List[np.ndarray], k: int) -> float:
        n_runs = len(H_runs)
        n_genes = H_runs[0].shape[1]
        similarities = []

        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                H_i = H_runs[i]
                H_j = H_runs[j]

                cost_matrix = np.zeros((k,k))
                for m1 in range(k):
                    for m2 in range(k):
                        cos_sim = self._cosine_similarity(H_i[m1, :], H_j[m2, :])
                        cost_matrix[m1, m2] = 1 - cos_sim

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                aligned_similarities = []
                for m1, m2 in zip(row_ind, col_ind):
                    sim = self._cosine_similarity(H_i[m1, :], H_j[m2, :])
                    aligned_similarities.append(sim)

                similarities.append(np.mean(aligned_similarities))
        return np.mean(similarities) if similarities else 0.0


    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return np.dot(a,b) / (norm_a * norm_b)

