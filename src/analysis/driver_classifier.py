import numpy as np
import pandas as pd
import logging
from typing import Tuple, List
from scipy.stats import binomtest

logger = logging.getLogger(__name__)

class DriverClassifier:
    def __init__(self, config: dict):
        cfg = config.get('driver_analysis', {})
        self.min_freq = cfg.get('min_mutation_frequency', 0.015)
        self.min_var = cfg.get('min_variance', 0.005)
        self.fdr_alpha = cfg.get('fdr_alpha', 0.05)
        self.use_stat_test = cfg.get('use_statistical_test', True)
        self.fallback_top_n = cfg.get('fallback_top_n', 50) # <-- НОВОЕ: спасительный fallback
        
        logger.info(f"DriverClassifier init (min_freq={self.min_freq}, FDR={self.fdr_alpha}, stat_test={self.use_stat_test})")

    def filter_drivers(self, X: np.ndarray, gene_names: List[str]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        logger.info(f"Analyzing {X.shape[1]} genes for driver status")

        forbidden = {'target', 'source_id', 'cluster_label', 'grade', 'age_years', 'sex', 'cancer_type', 'patient_id', 'sample_id'}
        
        stats = []
        driver_indices = []
        driver_names = []

        gene_freqs = np.array([np.nanmean(X[:, i]) for i in range(X.shape[1])])
        background_freq = np.nanmedian(gene_freqs[gene_freqs > 0]) if np.any(gene_freqs > 0) else 0.01
        logger.info(f"Background mutation frequency (median of non-zero): {background_freq:.4f}")

        p_values = []
        valid_indices = []

        for i, gene in enumerate(gene_names):
            if gene.lower() in forbidden:
                p_values.append(1.0)
                continue

            gene_col = X[:, i]
            true_freq = np.nanmean(gene_col)
            true_var = np.nanvar(gene_col)
            tested_count = int(np.sum(~np.isnan(gene_col)))
            mutated_count = int(np.nansum(gene_col))

            if tested_count > 0 and self.use_stat_test and mutated_count > 0:
                try:
                    result = binomtest(mutated_count, tested_count, background_freq, alternative='greater')
                    p_val = result.pvalue
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0

            p_values.append(p_val)
            valid_indices.append(i)

            stats.append({
                'Gene': gene,
                'Tested_N': tested_count,
                'Mutated_N': mutated_count,
                'True_Frequency': round(true_freq, 4),
                'Variance': round(true_var, 5),
                'P_Value': p_val
            })

        p_values_array = np.array(p_values)
        fdr_corrected = self._benjamini_hochberg(p_values_array)

        for idx, fdr in zip(valid_indices, fdr_corrected):
            if idx < len(stats):
                stats[idx]['FDR'] = round(float(fdr), 6)

        stats_df = pd.DataFrame(stats).sort_values(by='True_Frequency', ascending=False)

        for i, gene in enumerate(gene_names):
            if gene.lower() in forbidden or i >= len(stats):
                continue
            
            row = stats_df[stats_df['Gene'] == gene].iloc[0]
            freq_ok = row['True_Frequency'] >= self.min_freq
            var_ok = row['Variance'] >= self.min_var
            
            if self.use_stat_test:
                fdr_ok = row.get('FDR', 1.0) < self.fdr_alpha
                is_driver = freq_ok and var_ok and fdr_ok
            else:
                is_driver = freq_ok and var_ok

            if is_driver:
                driver_indices.append(i)
                driver_names.append(gene)

        if len(driver_names) == 0:
            logger.warning(f"Strict filtering found 0 drivers. Activating fallback: taking top {self.fallback_top_n} genes by variance.")
            stats_df_sorted_by_var = stats_df.sort_values(by='Variance', ascending=False)
            top_fallback = stats_df_sorted_by_var.head(self.fallback_top_n)
            
            driver_names = top_fallback['Gene'].tolist()
            driver_indices = [gene_names.index(g) for g in driver_names]
            logger.info(f"Fallback successful: selected {len(driver_names)} drivers.")

        logger.info(f"Driver filtering complete: {len(driver_names)} drivers selected out of {len(gene_names)} genes")
        if len(driver_names) > 0:
            logger.info(f"Top 5 drivers by frequency: {stats_df.head(5)['Gene'].tolist()}")

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
        
        for i in range(n - 2, -1, -1):
            fdr[i] = min(fdr[i + 1], sorted_p[i] * n / (i + 1))
        
        fdr = np.clip(fdr, 0, 1)
        
        result = np.zeros(n)
        result[sorted_indices] = fdr
        return result