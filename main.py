import warnings
import logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.utils.logger import setup_logger
from src.utils.paths import PathManager
from src.utils.config_loader import Config
from src.data.preprocessor import DataPreprocessor
from src.analysis.driver_classifier import DriverClassifier
from src.analysis.clusterer import PatientClusterer

def main(config_path: str = "config.yaml"):
    config = Config(config_path)
    paths = PathManager(config.config)
    paths.ensure_dirs()

    logger = setup_logger(name="CancerAnalyzer", log_file=str(paths.log_file), level=logging.INFO)
    logger.info("Starting pan-cancer analysis pipeline")

    logger.info("Stage 1 Data aggregation, normalization")
    preprocessor = DataPreprocessor(config.config)
    X_raw, y, gene_names, merged_df = preprocessor.load_and_process()

    if X_raw.size == 0:
        logger.error("Pipeline halted, no data loaded")
        return

    logger.info("Stage 2 driver vs passenger classification")
    driver_classifier = DriverClassifier(config.config)
    X_drivers, driver_names, driver_stats = driver_classifier.filter_drivers(X_raw, gene_names)

    if X_drivers.shape[1] < 5:
        logger.error("Pipeline halted too few drivers after filtering")
        return

    stats_path = paths.reports_dir / "driver_statistics.csv"
    driver_stats.to_csv(stats_path, index=False)
    logger.info(f"Driver statistics saved to {stats_path}")

    logger.info("Stage 3 unsupervised patient clustering")
    clusterer = PatientClusterer(config.config)
    cluster_labels, optimal_k, silhouette = clusterer.find_optimal_clusters(X_drivers)

    merged_df['cluster_label'] = cluster_labels

    cluster_report_path = paths.reports_dir / "cluster_assignments.csv"
    merged_df[['target','source_id', 'cluster_label']].to_csv(cluster_report_path, index=False)
    logger.info(f"Cluster assignments saved to {cluster_report_path}")

if __name__ == "__main__":
    main()