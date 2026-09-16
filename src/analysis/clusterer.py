import numpy as np
import pandas as pd
import logging
from typing import Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)

class PatientClusterer:
    def __init__(self, config:dict):
        cfg = config.get('clustering', {})
        self.min_clusters = cfg.get('min_clusters', 2)
        self.max_clusters = cfg.get('max_clusters', 6)
        self.random_state = cfg.get('random_state', 228)
        logger.info(f"PatientClusterer init (k range {self.min_clusters}-{self.max_clusters})")

    def find_optimal_clusters(self, X_drivers: np.ndarray) -> Tuple[np.ndarray, int, float]:
        logger.info("Starting unsupervised clustering")
        X_clean = np.nan_to_num(X_drivers, nan=0.0).astype(int)

        best_k = self.min_clusters
        best_score = -1.0
        best_labels = None

        for k in range(self.min_clusters, self.max_clusters + 1):
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                metric='jaccard',
                linkage='average'
            )

            labels = clusterer.fit_predict(X_clean)

            score = silhouette_score(X_clean, labels, metric='jaccard')

            logger.info(f"k = {k}, silhouette score {score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        logger.info(f"Optimal clustering found l={best_k} clusters with silhouette score {best_score:.4f}")

        unique, counts = np.unique(best_labels, return_counts=True)
        distribution = dict(zip([f"Cluster {i}" for i in unique], counts))
        logger.info(f"Cluster distr {distribution}")

        return best_labels, best_k, best_score