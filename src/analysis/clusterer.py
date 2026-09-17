import numpy as np
import pandas as pd
import logging
from typing import Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

class PatientClusterer:
    def __init__(self, config:dict):
        cfg = config.get('clustering', {})
        self.min_clusters = cfg.get('min_clusters', 2)
        self.max_clusters = cfg.get('max_clusters', 5)
        self.random_state = cfg.get('random_state', 228)

        self.split_large = cfg.get('split_large_clusters', True)
        self.max_size_to_split = cfg.get('max_cluster_size_to_split', 8000)
        self.min_split_k = cfg.get('min_split_clusters', 2)
        self.max_split_k = cfg.get('max_split_clusters', 4)
        self.max_samples = cfg.get('max_samples_for_base', 5000)

        logger.info(f"Clusterer init, hierarchial {self.split_large}, max_base_samples {self.max_samples}")

    def _cluster_subset(self, X_subset: np.ndarray, min_k: int, max_k: int) -> Tuple[np.ndarray, int, float]:
        n_samples = X_subset.shape[0]

        if n_samples <= self.max_samples:
            X_sample = X_subset
            sample_indices = np.arange(n_samples)
        else:
            rng = np.random.default_rng(self.random_state)
            sample_indices = rng.choice(n_samples, size=self.max_samples, replace=False)
            X_sample = X_subset[sample_indices]

        best_k = min_k
        best_score = -1.0
        best_sample_labels = None

        for k in range(min_k, max_k + 1):
            if k >= n_samples:
                break

        clusterer = AgglomerativeClustering(n_clusters=k, metric='jaccard', linkage='average')
        sample_labels = clusterer.fit_predict(X_sample)

        if len(np.unique(sample_labels)) > 1:
            score = silhouette_score(X_sample, sample_labels, metric='jaccard')
        else:
            score = -1.0

        if score > best_score:
            best_score = score
            best_k = k
            best_sample_labels = sample_labels

        knn = KNeighborsClassifier(n_neighbors=1, metric='jaccard')
        knn.fit(X_sample, best_sample_labels)
        full_labels = knn.predict(X_subset)

        return full_labels, best_k, best_score

    def find_optimal_clusters(self, X_drivers: np.ndarray) -> Tuple[np.ndarray, int, float]:
        logger.info("Starting unsupervised clustering")
        X_clean = np.nan_to_num(X_drivers, nan=0.0).astype(int)
        n_total = X_clean.shape[0]

        logger.info("Stage 1 base clustering of dataset")
        global_labels, base_k, base_score = self._cluster_subset(X_clean, self.min_clusters, self.max_clusters)
        logger.info(f"Base clustering found {base_k} clusters, score {base_score:.4f}")

        if self.split_large:
            logger.info("Stage 2 checking large clusters to split")
            unique_labels = np.unique(global_labels)
            new_cluster_id = max(unique_labels) + 1
            split_occurred = False

            for label in unique_labels:
                mask = (global_labels == label)
                cluster_size = np.sum(mask)

                if cluster_size > self.max_size_to_split:
                    logger.info(f"Cluster {label} large ({cluster_size} samples), splitting into {self.min_split_k}-{self.max_split_k} subclusters")
                    X_sub = X_clean[mask]

                    sub_labels, sub_k, sub_score = self._cluster_subset(X_sub, self.min_split_k, self.max_split_k)

                    cluster_indices = np.where(mask)[0] 

                    for i in range(sub_k):
                        sub_mask = (sub_labels == i)           
                        target_indices = cluster_indices[sub_mask] 
                        global_labels[target_indices] = new_cluster_id 
                        new_cluster_id += 1
                    split_occurred = True
                    logger.info(f"Successfully split cluster {label} into {sub_k} new subclusters")
                else:
                    logger.info(f"Cluster {label} size {cluster_size} is ok")

            if split_occurred:
                logger.info("Splitting complete evaluating distribution")
            else:
                logger.info("No clusters exceed the size for splitting")

        unique_final, counts = np.unique(global_labels, return_counts=True)
        label_map = {old: new for new, old in enumerate(unique_final)}
        final_labels = np.array([label_map[l] for l in global_labels])

        final_k = len(unique_final)
        distribution = dict(zip([f"Cluster {int(i)}" for i in unique_final], counts))

        logger.info(f"Final clustering found {final_k} distinct clusters")
        logger.info(f"Final cluster disribution {distribution}")

        return final_labels, final_k, base_score