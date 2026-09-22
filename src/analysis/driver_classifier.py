import numpy as np
import pandas as pd
import logging
from typing import Tuple, List
from scipy.stats import binomtest

logger = logging.getLogger(__name__)

class DriverClassifier:
    def __init__(self,config:dict):
        cfg = config.get('driver_analysis', {})
        self.min_freq = cfg.get('min_mutation_frequency',0.015)
        self.min_var = cfg.get('min_variance', 0.005)
        self.fdr_alpha = cfg.get('fdr_alpha', 0.05)
        self.use_stat_test = cfg.get('use_statistical_test', True)
        logger.info(f"DriverClassifier init (min_freq {self.min_freq}, FDR {self.fdr_alpha}, stat_test {self.use_stat_test})")

    def filter_drivers(self, X: np.ndarray, gene_names: List[str]) -> Tuple[np.ndarray, list[str], pd.DataFrame]:
        logger.info(f"Analyzing {X.shape[1]} genes for driver status")

        forbidden = {'target', 'source_id', 'cluster_label', 'grade', 'age_years', 'sex', 'cancer_type'}

        stats = []
        driver_indices = []
        driver_names = []

        gene_freqs = np.array([np.nanmean(X[:, i]) for i in range(X.shape[1])])
        backround_freq = np.nanmedian(gene_freqs)
        logger.info(f"Background mutation frequency {backround_freq:.4f}")

        p_values = []
        valid_indices = []

        for i, gene in enumerate(gene_names):
            if gene.lower() in forbidden:
                logger.warning(f"Skipping forbidden column {gene}")
                p_values.append(1.0)
                continue
            
            gene_col = X[:, i]
            true_freq = np.nanmean(gene_col)
            true_var = np.nanvar(gene_col)
            tested_count = int(np.sum(~np.isnan(gene_col)))
            mutated_count = np.nansum(gene_col)

            if tested_count > 0 and self.use_stat_test:
                try:
                    result = binomtest(mutated_count, tested_count, backround_freq, alternative='greater')
                    p_val = result.pvalue
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0

            p_values.append(p_val)
            valid_indices.append(i)

            stats.append({
                'Gene': gene,
                'Tested_N': int(tested_count),
                'Mutated_N': int(mutated_count),
                'True_Frequency': round(true_freq, 4),
                'Variance': round(true_var, 5)
            })

        p_values_array = np.array(p_values)
        fdr_corrected = self._benjamini_hochberg(p_values_array)

        for idx, fdr in zip(valid_indices, fdr_corrected[valid_indices] if len(valid_indices) == len(fdr_corrected) else fdr_corrected):
            if idx < len(len(stats)):
                stats[idx]['FDR'] = round(fdr, 6)

        status_df = pd.DataFrame(stats).sort_values(by='True_Frequency', ascending=False)

        

        logger.info(f"Driver filtering complete {len(driver_names)} drivers selected out of {len(gene_names)} genes")
        if len(driver_names) > 0:
            logger.info(f"Top 5 drivers by frequency {stats_df.head(5)['Gene'].tolist()}")
        else:
            logger.warning('no driver found')
        X_drivers = X[:, driver_indices]

        return X_drivers, driver_names, stats_df

    def _benjamini_hochberg(self, p_values: np.ndarray) -> np.ndarray:
        n = len(p_values)
        if n == 0:
            return np.array([])

        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        fdr = np.zeros(n)
        fdr[-1] = sorted_p[-1]

        for i in range(n-2, -1, -1):
            fdr[i] = min(fdr[i+1],sorted_p[i] * n / (i+1))

        fdr = np.clip(fdr, 0, 1)

        result = np.zeros(n)
        result[sorted_indices] = fdr
        return result