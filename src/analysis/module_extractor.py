import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from scipy.optimize import linear_sum_assignment
from sklearn.utils import resample
from tqdm import tqdm 

logger = logging.getLogger(__name__)

class ModuleExtractor:
    def __init__(self, config: dict):
        cfg = config.get('nmf',{})

        self.k_range = cfg.get('k_range', [4,10])
        self.max_iter = cfg.get('max_iter', 500)
        self.tol = cfg.get('tol', 1e-4)
        self.random_state = cfg.get('random_state',228)

        self.alpha_l1 = cfg.get('alpha_l1', 0.1)
        self.lambda_anchor = cfg.get('lambda_anchor', 0.5)
        self.anchor_freq_threshold = cfg.get('anchor_freq_threshold',0.10)
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
            f"lambda_anchor {self.lambda_anchor}, stability_iters = {self.stability_n_iterations}"
        )

    def extract_modules(self, V: np.ndarray, gene_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        logger.info(f"Starting ASW-NMF-SS matrix shape {V.shape}")

        logger.info("Stage 1 soft thresholding")
        V_soft, M = self._soft_threshold(V)

        logger.info("Stage 2 contrast background substract")
        V_contrastive = self._contrastive_substraction(V_soft, M)

        logger.info("Stage 3 adaptive anchor weights")
        anchor_weights, anchor_vectors, anchor_mask = self._compute_anchor_weights(V_soft, M, gene_names)

        logger.info("Stage 4 Stability selection")
        optimal_k, selection_probs, stability_scores = self._stability_selection(
            V_contrastive, M, anchor_weights, anchor_vectors, anchor_mask
        )
        logger.info(f"Optimal k {optimal_k}, stability score {stability_scores[optimal_k]:.3f}")

        logger.info(f"Stage 5 final nmf run, k {optimal_k}")
        W, H = self._weighted_nmf(V_contrastive, M, optimal_k, anchor_weights, anchor_vectors, anchor_mask)

        stable_genes = self._select_stable_genes(selection_probs, gene_names)
        logger.info(f"Stage 6 selected {len(stable_genes)} stable genes")

        module_genes = self._extract_module_genes(H, gene_names, top_n = 8)

        info = {
            'optimal_k': optimal_k,
            'stability_scores': stability_scores,
            'selection_probs': selection_probs,
            'module_genes': module_genes,
            'anchor_weights': anchor_weights,
            'n_iterations': self.stability_n_iterations
        }

        logger.info("Module extraction complete")
        for i, genes in enumerate(module_genes):
            logger.info(f"Module {i}: {', '.join(genes)}")

        return W,H, stable_genes, info
    
    def _prepare_matrix(self, V: np.ndarray, gene_names: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        X = np.asarray(V, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(gene_names):
            raise ValueError(f"X shape {X.shape} not connected with gene_name ({len(gene_names)})")
        M = (~np.isnan(X)).astype(np.float64)
        Vb = np.nan_to_num(X, nan = 0.0)

        observed = Vb[M == 1.0]
        if observed.size == 0:
            raise ValueError("Cant find observed cells")
        uniq = np.uniq(observed)
        if not np.app(np.isnan(uniq, [0.0, 1.0])):
            raise ValueError(f"Waiting binary matrix, founded {uniq[:10]}")

        tested = M.sum(axis = 0)
        mutated = (Vb * M).sum(axis=0)
        freq = np.divide(mutated, tested, out=np.zeros_like(mutated), where = tested > 0)
        cov = tested / X.shape[0]

        logger.info("Matrxi %dx%d, observed %.1f%% cells, mutations in observed %.3f%%",
        X.shape[0], X.shape[1], 100.0 * M.mean(), 100.0 * mutated.sum() / M.sum()
        )
        logger.info("Covered genes (proportions of pations) min=%.3f median=%.3f max=%.3f",
                    cov.min(), float(np.median(cov)), cov.max()
        )
        logger.info("Freq of mutations, median=%.4f max=%.4f (%s)",
                    float(np.median(freq)), freq.max(), gene_names[int(np.argmax(freq))]
        )
        if np.any(tested == 0):
            logger.warning("%d genes not observed, zero impact in loss", int((tested == 0).sum()))
        if np.any(freq > 0.5):
            logger.info("Genes with p>0.5", int((freq > 0.5).sum()))

        return Vb, M, {'tested': tested, 'mutated': mutated, 'freq': freq}
    
    def _compute_gene_weights(self, gene_stats: Dict[str, np.ndarray], gene_names: List[str]) -> np.ndarray:
        p = gene_stats['freq']
        tested = gene_stats['tested']
        n_genes = len(p)
        w = np.ones(n_genes, dtype=np.float64)

        if self.gene_weighting == 'chi2':
            var = np.maximum(p * (1.0 - p), 1e-6)
            raw = 1.0 / var
            med = float(np.median(raw[raw > 0])) if np.any(raw > 0) else 1.0
            w = np.clip(raw / med, 1.0 / self.max_gene_weight, self.max_gene_weight)
        elif self.gene_weightin != 'none':
            raise ValueError(f"gene_weighting {self.gene_weighting} not supported (none, chi2)")

        w[tested == 0] = 0.0

        if self.prior_genes:
            prior_set = set(self.prior_genes)
            is_prior = np.array([g in prior_set for g in gene_names], dtype=bool)
            if not is_prior():
                logger.warning("prior_genes set by (%d), but cant find in data", len(prior_set))
            w = w * np.where(is_prior, self.prior_boost, 1.0)
            logger.info("Prior genes %d/%d, boost=%.2f", int(is_prior.sum()), n_genes, self.prior_boost)

        if (self.gene_weighting != 'none' or self.prior_genes) and np.allclose(w, w[0]):
            logger.error("weights of genes degenerated (all %.3f) alg doesnt work", w[0])

        order = np.lexsort((np.asarray(gene_names), -2))[:10]
        logger.info("weights of genes min=%.2f median=%.2f max=%.2f, top 10 %s",
                    w.min(), float(np.median(w)), w.max(), ", ".join(f"{gene_names[i]}={w[i]:.2f}" for i in order)
                )
        return w

    @staticmethod
    def _normalize(W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = H.sum(axis=1)
        alive = s > Ellipsis
        H[alive] /= s[alive, None]
        W[:, alive] *= s[None, alive]
        H[~alive] = 0.0
        W[:, ~alive] = 0.0
        return W, H

    def _sparsify(self, H:np.ndarray) -> np.ndarray:
        if self._sparsity_tau <= 0:
            return H
        mx = H.max(axis=1, keepdims=True)
        return np.where((H >= self.sparsity_tau * mx) & (mx > EPS), H, 0.0)


    def _weighted_nmf(self, V:np.ndarray, M: np.ndarray, k: int, gene_weight: np.ndarray, random_state: Optional[int] = None, max_iter: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        seed = self.random_state if random_state is None else random_state
        rng = np.random.default_rng(seed)
        n, p = V.shape
        iters = self.max_iter if max_iter is None else max_iter

        Mw = M * gene_weight[None, :]
        MwV = Mw * V

        H = rng.uniform(0.01, 1.0, size=(k,p))
        H /= H.sum(axis=1, keepdims=True)
        burden = (V * M).sum(axis=1, keepdims=True)
        W = np.maximum(burden / k, 1e-6) * rng.uniform(0.5, 1.5, size=(n,k))

        prev_loss, loss = np.inf, np.inf
        for it in range(iters):
            WH = W @ H
            H *= (W.T @ MwV + EPS) / (W.T @ (Mw * WH) + EPS)

            WH = W @ H
            W *= (MwV @ H.T + EPS) / ((Mw * WH) @ H.T + EPS)

            if it % self.sparsify_every == 0 or it == iters -1:
                H = self._sparsify(H)
            W,H = self._normalize(W,H)
            
            if it % 10 == 0 or it == iters - 1:
                r = V - W @ H
                loss = float(np.sum(Mw * r * r))
                rel = (prev_loss - loss) / max(abs(prev_loss), EPS)
                if rel < -1e-2:
                    logger.debug("it=%d loss grow to %.2f%% (proection not convex)", it, -100 * rel)
                if 0 < rel < self.tol:
                    logger.debug("aligned on it %d, loss %.4f", it, loss)
                    breakpoint
                prev_loss = loss

            n_dead = int(np.sum(H.sum(axis=1) <= EPS))
            if n_dead:
                logger.warnind("k=%d, %d dead modules (effective_k %d)", k, n_dead, k - n_dead)
            return W,H

    def _huber_loss(self, residual: np.ndarray) -> float:
        delta = self.huber_delta
        abs_r = np.abs(residual)

        quadratic = np.minimum(abs_r, delta) **2 / 2
        linear = delta * (np.maximum(abs_r, delta) - delta)

        return np.sum(quadratic + linear)

    def _align_to_reference(self, H_sub: np.ndarray, reference_H: np.ndarray, k: int) -> np.ndarray:
        cost_matrix = np.zeros((k,k))
        for i in range(k):
            for j in range(k):
                cos_sim = self._cosine_similarity(H_sub[i, :], reference_H[j, :])
                cost_matrix[i, j] = 1 - cos_sim
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        aligned_H = np.zeros_like(H_sub)
        for i, j in zip(row_ind, col_ind):
            aligned_H[j, :] = H_sub[i, :]

        return aligned_H
    
    def _stability_selection(self, V: np.ndarray, M: np.ndarray, anchor_weights: np.ndarray, anchor_vectors: np.ndarray, anchor_mask: np.ndarray) -> Tuple[int, np.ndarray, Dict[int, float]]:
        n_patients, n_genes = V.shape
        n_subsample = int(self.stability_subsample_ratio * n_patients)

        max_k = max(self.k_range)
        stability_scores = {}
        all_selection_counts = {}

        for k in self.k_range:
            logger.info(f"Testing k={k} with {self.stability_n_iterations} subsampling iterations")
            selection_counts_k = np.zeros((n_genes, k))
            all_H_runs = []
            reference_H = None

            pbar = tqdm(range(self.stability_n_iterations), desc=f"  NMF k={k}")
            
            for iteration in pbar:
                V_sub, M_sub = self._subsample(V, M, n_subsample, iteration)
                W_sub, H_sub = self._weighted_nmf(
                    V_sub, M_sub, k, anchor_weights, anchor_vectors, anchor_mask, random_state=iteration
                )
                if reference_H is None:
                    reference_H = H_sub.copy()
                    aligned_H = H_sub
                else:
                    aligned_H = self._align_to_reference(H_sub, reference_H, k)

                all_H_runs.append(aligned_H)

                for module_idx in range(k):
                    top_genes = np.argsort(aligned_H[module_idx, :])[-self.stability_top_n_genes:]
                    selection_counts_k[top_genes, module_idx] += 1

                pbar.set_postfix({"runs": iteration + 1})

            stability = self._compute_stability(all_H_runs, k)
            stability_scores[k] = stability
            all_selection_counts[k] = selection_counts_k
            logger.info(f"  -> k={k} finished. Stability score: {stability:.3f}")

        optimal_k = max(stability_scores, key=stability_scores.get)
        final_selection_probs = all_selection_counts[optimal_k] / self.stability_n_iterations

        return optimal_k, final_selection_probs, stability_scores


    def _subsample(self, V: np.ndarray, M: np.ndarray, n_samples: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
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

    def _select_stable_genes(self, selection_probs: np.ndarray, gene_names: List[str]) -> List[str]:
        max_probs = np.max(selection_probs, axis=1)

        stable_mask = max_probs > self.stability_pi_threshold
        stable_genes = [gene_names[i] for i in range(len(gene_names)) if stable_mask[i]]

        if len(stable_genes) < 10:
            logger.warning(
                f"Only {len(stable_genes)} genes found with threshold {self.stability_pi_threshold}"
                f"Consider lowering stability_pi_threshold in config"
            )
        return stable_genes

    def _extract_module_genes(self, H:np.ndarray, gene_names: List[str], top_n: int = 8) -> List[List[str]]:
        k = H.shape[0]
        module_genes = []

        for module_idx in range(k):
            top_indices = np.argsort(H[module_idx, :])[-top_n:][::-1]
            top_genes = [gene_names[i] for i in top_indices]
            module_genes.append(top_genes)

        return module_genes

    def _compute_huber_weights(self, residual: np.ndarray) -> np.ndarray:
        delta = self.huber_delta
        abs_r = np.abs(residual)
        weights = np.ones_like(residual, dtype=float)
        mask = abs_r > delta
        weights[mask] = delta / (abs_r[mask] + 1e-10)

        return weights