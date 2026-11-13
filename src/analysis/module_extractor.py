import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from scipy.optimize import linear_sum_assignment
from sklearn.utils import resample
from tqdm import tqdm 

logger = logging.getLogger(__name__)
EPS = 1e-10

class ModuleExtractor:
    def __init__(self, config: dict):
        cfg = config.get('nmf', {})
        self.k_values = self._resolve_k_values(cfg)
        self.max_iter = int(cfg.get('max_iter', 500))
        self.tol = float(cfg.get('tol', 1e-5))
        self.random_state = int(cfg.get('random_state', 228))

        self.gene_weighting = cfg.get('gene_weighting', 'none')
        self.prior_genes = list(cfg.get('prior_genes', []))
        self.prior_boost = float(cfg.get('prior_boost', 2.0))
        self.max_gene_weight = float(cfg.get('max_gene_weight', 5.0))

        self.sparsity_tau = float(cfg.get('sparsity_tau', 0.05))
        self.sparsify_every = int(cfg.get('sparsify_every', 5))

        self.stability_n_iterations = int(cfg.get('stability_n_iterations', 25))
        self.stability_subsample_ratio = float(cfg.get('stability_subsample_ratio', 0.5))
        self.stability_max_iter = int(cfg.get('stability_max_iter', 150))
        self.stability_pi_threshold = float(cfg.get('stability_pi_threshold', 0.7))
        self.stability_top_n_genes = int(cfg.get('stability_top_n_genes', 10))
        self.top_n_module_genes = int(cfg.get('top_n_module_genes', 8))

        for dead in ('soft_thresholding', 'lambda_anchor', 'anchor_freq_threshold',
                     'anchor_var_threshold', 'alpha_l1', 'huber_delta', 'k_range'):
            if dead in cfg:
                logger.warning("nmf.%s old and unused", dead)

    def extract_modules(self, V: np.ndarray, gene_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        logger.info("Masked weighted NMF (simplex-H, sparse), input %s", V.shape)

        Vb, M, gstats = self._prepare_matrix(V, gene_names)
        g = self._compute_gene_weights(gstats, gene_names)

        best_k, sel_probs, scores, details = self._stability_selection(Vb, M, g)
        logger.info("Selected k=%d (stability=%.3f)", best_k, scores[best_k])

        W, H = self._weighted_nmf(Vb, M, best_k, g)
        W, H = self._normalize(W, H)

        r = Vb - W @ H
        rmse = float(np.sqrt(np.sum(M * r * r) / max(M.sum(), 1.0)))
        logger.info(
            "Final : k=%d, RMSE(tested)=%.4f, dead modules=%d, "
            "slow load W=%.3f",
            best_k, rmse, int(np.sum(H.sum(axis=1) <= EPS)), float(np.median(W.sum(axis=1)))
        )

        stable_genes = self._select_stable_genes(sel_probs, gene_names)
        module_genes = self._extract_module_genes(H, gene_names, self.top_n_module_genes)
        for i, genes in enumerate(module_genes):
            logger.info("Module %d: %s", i, ", ".join(genes))

        info = {
            'optimal_k': best_k,
            'stability_scores': scores,
            'stability_details': details,
            'selection_probs': sel_probs,
            'module_genes': module_genes,
            'gene_weights': g,
            'rmse_tested': rmse,
            'effective_k': details[best_k]['effective_k'],
            'n_iterations': self.stability_n_iterations,
            'objective': 'masked_weighted_sse_simplex_sparse',
        }
        return W, H, stable_genes, info
    
    def _prepare_matrix(self, V: np.ndarray, gene_names: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        X = np.asarray(V, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(gene_names):
            raise ValueError(f"X shape {X.shape} not connected with gene_name ({len(gene_names)})")
        M = (~np.isnan(X)).astype(np.float64)
        Vb = np.nan_to_num(X, nan = 0.0)

        observed = Vb[M == 1.0]
        if observed.size == 0:
            raise ValueError("Cant find observed cells")
        uniq = np.unique(observed)
        if not np.all(np.isin(uniq, [0.0, 1.0])):
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
            logger.info("Genes with p>0.5, %d", int((freq > 0.5).sum()))

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
        elif self.gene_weighting != 'none':
            raise ValueError(f"gene_weighting {self.gene_weighting} not supported (none, chi2)")

        w[tested == 0] = 0.0

        if self.prior_genes:
            prior_set = set(self.prior_genes)
            is_prior = np.array([g in prior_set for g in gene_names], dtype=bool)
            if not is_prior.any():
                logger.warning("prior_genes set by (%d), but cant find in data", len(prior_set))
            w = w * np.where(is_prior, self.prior_boost, 1.0)
            logger.info("Prior genes %d/%d, boost=%.2f", int(is_prior.sum()), n_genes, self.prior_boost)

        if (self.gene_weighting != 'none' or self.prior_genes) and np.allclose(w, w[0]):
            logger.error("weights of genes degenerated (all %.3f) alg doesnt work", w[0])

        order = np.lexsort((np.asarray(gene_names), -w))[:10]
        logger.info("weights of genes min=%.2f median=%.2f max=%.2f, top 10 %s",
                    w.min(), float(np.median(w)), w.max(), ", ".join(f"{gene_names[i]}={w[i]:.2f}" for i in order)
                )
        return w

    @staticmethod
    def _normalize(W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = H.sum(axis=1)
        alive = s > EPS
        H[alive] /= s[alive, None]
        W[:, alive] *= s[None, alive]
        H[~alive] = 0.0
        W[:, ~alive] = 0.0
        return W, H

    def _sparsify(self, H:np.ndarray) -> np.ndarray:
        if self.sparsity_tau <= 0:
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
                if 0 <= rel < self.tol:
                    logger.debug("aligned on it %d, loss %.4f", it, loss)
                    break
                prev_loss = loss

        n_dead = int(np.sum(H.sum(axis=1) <= EPS))
        if n_dead:
            logger.warning("k=%d, %d dead modules (effective_k %d)", k, n_dead, k - n_dead)
        return W,H

    @staticmethod
    def _resolve_k_values(cfg: dict) -> List[int]:
        if 'k_values' in cfg:
            ks = sorted({int(k) for k in cfg['k_values']})
        else:
            lo, hi = cfg.get('k_range', [4,10])
            ks = list(range(int(lo), int(hi) +1))
            logger.warning("nmf.k_range [%s,%s] is closed range %s", lo, hi, ks)
        if not ks or min(ks) < 2:
            raise ValueError(f"Incorrect grid k {ks}")
        return ks

    @staticmethod
    def _pairwise_similarity(runs: List[np.ndarray]) -> Tuple[np.ndarray, Dict]:
        R = len(runs)
        Hn = [r/ np.maximum(np.linalg.norm(r, axis=1, keepdims=True), EPS) for r in runs]
        S = np.eye(R)
        perms: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for i in range(R):
            for j in range(i + 1, R):
                cost = 1.0 - Hn[i] @ Hn[j].T
                ri, ci = linear_sum_assignment(cost)
                S[i, j] = S[j, i] = float(np.mean(1.0 - cost[ri, ci]))
                perms[(i,j)] = (ri, ci)
        return S, perms

    @staticmethod
    def _apply_perm(H_r: np.ndarray, perms: Dict, r: int, ref: int) -> np.ndarray:
        if r == ref:
            return H_r
        out = np.zeros_like(H_r)
        if (r, ref) in perms:
            ri,ci = perms[(r,ref)]
            out[ci] = H_r[ri]
        else:
            ri, ci = perms[(ref,r)]
            out[ri] = H_r[ci]
        return out
    
    def _stability_selection(self, V: np.ndarray, M: np.ndarray, gene_weight: np.ndarray) -> Tuple[int, np.ndarray, Dict[int, float], Dict[int,Dict]]:
        n, p = V.shape
        n_sub = max(2, int(self.stability_subsample_ratio * n))
        rng = np.random.default_rng(self.random_state)

        scores: Dict[int, float] = {}
        details: Dict[int, Dict] = {}
        sel: Dict[int, np.ndarray] = {}

        for k in self.k_values:
            runs: List[np.ndarray] = []
            pbar = tqdm(range(self.stability_n_iterations), desc=f"NMF k={k}", leave=False)
            for _ in pbar:
                sub_seed = int(rng.integers(0, 2**31-1))
                init_seed = int(rng.integers(0, 2**31 -1))
                idx = np.random.default_rng(sub_seed).choice(n,size=n_sub,replace=False)
                _, H_s = self._weighted_nmf(
                    V[idx], M[idx], k, gene_weight,
                    random_state=init_seed, max_iter=self.stability_max_iter
                )
                runs.append(H_s)

            S, perms = self._pairwise_similarity(runs)
            iu = np.triu_indices(len(runs), 1)
            mean_sim = float(S[iu].mean()) if len(runs) > 1 else 0.0
            per_run = S.sum(axis=1) / max(len(runs) - 1, 1)
            se = float(per_run.std(ddof=1) / np.sqrt(len(runs))) if len(runs) > 1 else 0.0

            medoid = int(np.argmax(S.sum(axis=1)))
            counts = np.zeros((p,k))
            for r, H_r in enumerate(runs):
                H_al = self._apply_perm(H_r, perms, r, medoid)
                for m in range(k):
                    row = H_al[m]
                    nz = np.flatnonzero(row > 0)
                    if nz.size == 0:
                        continue
                    top = nz[np.lexsort((nz, -row[nz]))[:self.stability_top_n_genes]]
                    counts[top, m] += 1

            eff_k = float(np.mean([np.count_nonzero(Hr.sum(axis=1) > EPS) for Hr in runs]))
            scores[k] = mean_sim
            details[k] = {'mean': mean_sim, 'se_runs': se, 'effective_k': eff_k}
            sel[k] = counts / self.stability_n_iterations
            logger.info("k=%d, stability=%.3f +- %.3f (SE), effective_k=%.1f", k, mean_sim, se, eff_k)

        best_k = max(scores, key=scores.get)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if len(ranked) > 1:
            (k1, s1), (k2, s2) = ranked[0], ranked[1]
            se_pair = float(np.hypot(details[k1]['se_runs'], details[k2]['se_runs']))
            if (s1 -s2) < 2 * se_pair:
                logger.warning("k doesnt identify, k=%d (%.3f) vs k=%d (%.3f), gap=%.3f < 2·SE=%.3f",
                               k1, s1, k2, s2, s1 -s2, 2 * se_pair
                )
        return best_k, sel[best_k], scores, details
    
    def _select_stable_genes(self, probs: np.ndarray, gene_names: List[str]) -> List[str]:
        mx = probs.max(axis=1)
        stable = [gene_names[i] for i in np.flatnonzero(mx > self.stability_pi_threshold)]
        chance = min(1.0, probs.shape[1] * self.stability_top_n_genes / len(gene_names))
        logger.info("stability genes %d/%d (pi>%.2f) marginal chance get in top-N ~= %.2f",
                    len(stable), len(gene_names), self.stability_pi_threshold, chance
                    )
        if self.stability_pi_threshold <= chance:
            logger.warning("threshold pi is not higher than random level, stability incorrect")
        return stable
    
    def _extract_module_genes(self, H:np.ndarray, gene_names: List[str], top_n: int = 8) -> List[List[str]]:
        modules = []
        for m in range(H.shape[0]):
            row = H[m]
            nz = np.flatnonzero(row >0)
            top = nz[np.lexsort((nz, -row[nz]))[:top_n]] if nz.size else np.array([], dtype=int)
            modules.append([gene_names[i] for i in top])
        from collections import Counter
        shared = {gn: c for gn, c in Counter(x for mod in modules for x in mod).items() if c > 1}
        if shared:
            logger.info("Genes in >1 module (%d), %s", len(shared), shared)
        return modules

