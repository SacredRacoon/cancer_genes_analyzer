import numpy as np
import pandas as pd
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)
class DriverClassifier:
    def __init__(self,config:dict):
        cfg = config.get('driver_analysis', {})
        self.min_freq = cfg.get('min_mutation_frequency',0.015)
        self.min_var = cfg.get('min_variance', 0.005)
        logger.info(f"DriverClassifier init (min_freq {self.min_freq}, min_var {self.min_var})")

    def filter_drivers(self, X: np.ndarray, gene_names: List[str]) -> Tuple[np.ndarray, list[str], pd.DataFrame]:
        logger.info(f"Analyzing {X.shape[1]} genes for driver status")

        stats = []
        driver_indices = []
        driver_names = []

        for i, gene in enumerate(gene_names):
            gene_col = X[:, i]

            true_freq = np.nanmean(gene_col)
            true_var = np.nanvar(gene_col)

            tested_count = np.sum(~np.isnan(gene_col))
            mutated_Count = np.nansum(gene_col)

            stats.append({
                'Gene': gene,
                'Tested_N': int(tested_count),
                'Mutated_N': int(mutated_Count),
                'True_Frequency': round(true_freq, 4),
                'Variance': round(true_var, 5)
            })

            if true_freq > self.min_freq and true_var >= self.min_var:
                driver_indices.append(i)
                driver_names.append(gene)

            stats_df = pd.DataFrame(stats).sort_values(by='True_Frequency', ascending=False)

            logger.info(f"Driver filtering complete {len(driver_names)} drivers selected out of {len(gene_names)} genes")
            logger.info(f"Top 5 drivers by frequency {stats_df.head(5)['Gene'].tolist()}")

            X_drivers = X[:, driver_indices]

            return X_drivers, driver_names, stats_df