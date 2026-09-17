import numpy as np
import pandas as pd
import logging
from itertools import combinations
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class SignatureMiner:
    def __init__(self, config: dict):
        cfg = config.get('signature_mining', {})
        self.min_diff = cfg.get('min_frequency_difference', 0.10)
        self.min_support = cfg.get('min_support_count', 15)
        self.max_combo_size = cfg.get('max_combination_size', 3)
        logger.info(f"SignatureMiner init (min_diff{self.min_diff}, max_size{self.max_combo_size})")

    def mine_signatures(self, X_drivers: np.ndarray, cluster_labels: np.ndarray, driver_names: List[str]) -> Dict[int, pd.DataFrame]:
        logger.info("Starting signature mining")
        unique_clusters = np.unique(cluster_labels)
        all_signatures = {}

        X_binary = np.nan_to_num(X_drivers, nan=0.0).astype(bool)
        total_patients = X_binary.shape[0]

        for cluster in unique_clusters:
            logger.info(f"Mining signatures for сluster {cluster}")
            mask_target = (cluster_labels == cluster)
            mask_other = ~mask_target
            
            n_target = np.sum(mask_target)
            n_other = np.sum(mask_other)
            
            if n_target < self.min_support:
                logger.warning(f"Cluster {cluster} too small ({n_target}), skipping")
                continue

            cluster_sigs = []

            for size in range(2, self.max_combo_size + 1):
                target_freqs = np.mean(X_binary[mask_target], axis=0)
                candidate_indices = np.where(target_freqs >= 0.05)[0]
                
                if len(candidate_indices) < size:
                    continue

                candidate_names = [driver_names[i] for i in candidate_indices]
                total_combos = len(list(combinations(range(len(candidate_indices)),size)))

                logger.info(f"Size {size}, testing {total_combos} combinations")
                
                for combo in combinations(range(len(candidate_indices)), size):
                    actual_indices = [candidate_indices[i] for i in combo]
                    gene_names = [candidate_names[i] for i in combo]
                    
                    has_combo = np.all(X_binary[:, actual_indices], axis=1)
                    
                    support_target = np.sum(has_combo[mask_target])
                    support_other = np.sum(has_combo[mask_other])
                    
                    freq_target = support_target / n_target
                    freq_other = support_other / n_other if n_other > 0 else 0.0
                    diff = freq_target - freq_other
                    
                    if diff >= self.min_diff and support_target >= self.min_support:
                        cluster_sigs.append({
                            'Signature': ' + '.join(gene_names),
                            'Genes': gene_names,
                            'Size': size,
                            'Freq_Target_%': round(freq_target * 100, 2),
                            'Freq_Other_%': round(freq_other * 100, 2),
                            'Difference_%': round(diff * 100, 2),
                            'Support_Target': int(support_target),
                            'Support_Other': int(support_other)
                        })
            
            if cluster_sigs:
                df = pd.DataFrame(cluster_sigs).sort_values(by='Difference_%', ascending=False)
                all_signatures[int(cluster)] = df
                logger.info(f"Found {len(df)} valid signatures for cluster {cluster}")
            else:
                logger.info(f"No valid signatures found for cluster {cluster}")

        return all_signatures