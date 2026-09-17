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
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import ResultVisualizer
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
        logger.info("Building binary signature matrix for GA")
        X_binary = np.nan_to_num(X_drivers, nan=0.0).astype(bool)
        X_sigs = np.zeros((X_binary.shape[0], len(signature_list)), dtype=int)

        for sig_idx, sig_str in enumerate(signature_list):
            genes_in_sig = [g.strip() for g in sig_str.split('+')]
            gene_indices = [driver_names.index(g) for g in genes_in_sig if g in driver_names]

            if len(gene_indices) == len(genes_in_sig):
                X_sigs[:, sig_idx] = np.all(X_binary[:, gene_indices], axis=1).astype(int)
        X_ga = X_sigs
        ga_feature_names = signature_list
        logger.info(f"Successfully mapped {len(signature_list)}")
    else:
        logger.warning("No signatures found, back to top 30 drivers")
        X_ga = X_drivers[:, :30]
        ga_feature_names = driver_names[:30]

    logger.info("Stage 5 Coevolutionary GA")
    ga = SignatureGeneticSelector(config.config, X_ga, y, ga_feature_names)
    best_chromosome, best_fitness, best_features = ga.run(verbose=True)


    logger.info("Stage 6 reporting")
    selected_indices = np.where(best_chromosome == 1)[0]

    evaluator = ModelEvaluator(config.config, paths)
    eval_results = evaluator.evaluate_and_save(X_ga, y, selected_indices, ga_feature_names)

    visualizer = ResultVisualizer(paths)
    visualizer.plot_evolution(ga.history)
    visualizer.plot_feature_importance(eval_results['importance_df'])

    cluster_names = [f"Cluster_{i}" for i in sorted(np.unique(y))]
    visualizer.plot_confusion_matrix(eval_results['y_test'],eval_results['y_pred'])

    reporter = PipelineReporter(paths)
    reporter.generate_final_report(
        y_test=eval_results['y_test'],
        y_pred=eval_results['y_pred'],
        y_proba=eval_results['y_proba'],
        cluster_names=cluster_names,
        best_signatures=best_features,
        signature_importance=eval_results['importance_df']
    )   


if __name__ == "__main__":
    main()