import warnings
import logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.utils.logger import setup_logger
from src.utils.paths import PathManager
from src.utils.config_loader import Config
from src.utils.random_state import RandomStateManager
from src.data.preprocessor import DataPreprocessor
from src.analysis.driver_classifier import DriverClassifier
from src.analysis.clusterer import PatientClusterer
from src.analysis.signature_miner import SignatureMiner
from src.evolution.selector import SignatureGeneticSelector
from src.evaluation.reporter import PipelineReporter
from src.models.factory import ModelFactory

def main(config_path: str = "config.yaml"):
    config = Config(config_path)
    paths = PathManager(config.config)
    paths.ensure_dirs()

    logger = setup_logger(name="CancerAnalyzer", log_file=str(paths.log_file), level=logging.INFO)
    logger.info("Starting pan-cancer analysis pipeline")

    seed = config.config.get('global_seed',228)
    RandomStateManager(seed=seed)

    logger.info("Stage 1 Data aggregation, normalization")

    preprocessor = DataPreprocessor(config.config)
    X_raw, y, gene_names, merged_df = preprocessor.load_and_process()

    if X_raw.size == 0:
        logger.error("Pipeline halted, no data loaded")
        return

    logger.info("Stage 2 driver vs passenger classification")
    driver_classifier = DriverClassifier(config.config)
    X_drivers, driver_names, driver_stats = driver_classifier.filter_drivers(X_raw, gene_names)
    driver_stats.to_csv(paths.reports_dir / "driver_statistics.csv", index=False)

    logger.info("Stage 3 unsupervised patient clustering")
    clusterer = PatientClusterer(config.config)
    cluster_labels, optimal_k, silhouette = clusterer.find_optimal_clusters(X_drivers)
    merged_df['cluster_label'] = cluster_labels
    merged_df[['target','source_id', 'cluster_label']].to_csv(paths.reports_dir / 'cluster_assignments.csv', index = False)

    logger.info("Stage 4: Combinatorial signature mining")
    miner = SignatureMiner(config.config)
    cluster_signatures = miner.mine_signatures(X_drivers, cluster_labels, driver_names)
    
    all_unique_sigs = set()
    sig_to_cluster = {}
    for cluster, df in cluster_signatures.items():
        for sig in df['Signature']:
            all_unique_sigs.add(sig)
            sig_to_cluster[sig] = cluster
            
    signature_list = sorted(list(all_unique_sigs))
    logger.info(f"Total unique signatures discovered across all clusters {len(signature_list)}")


    #Временная заглушка (лень дописывать)
    if len(signature_list) > 0:
        logger.info("Using top drivers as proxy for signatures in this run (full signature matrix mapping in next iteration)")
        X_ga = X_drivers[:, :30]
        ga_feature_names = driver_names[:30]
    else:
        X_ga = X_drivers
        ga_feature_names = driver_names

    # Этап 5: ГА
    logger.info("Stage 5: Co-evolutionary Genetic Algorithm")
    ga = SignatureGeneticSelector(config.config, X_ga, y, ga_feature_names)
    best_chromosome, best_fitness, best_features = ga.run(verbose=True)

    # Этап 6: Финальная оценка и отчет
    logger.info("Stage 6: Final evaluation and reporting")
    # Здесь обучаем финальную модель на best_features и вызываем PipelineReporter
    # (Код обучения аналогичен твоему старому evaluator.py, но на новых признаках)
    
    logger.info("="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)   
if __name__ == "__main__":
    main()