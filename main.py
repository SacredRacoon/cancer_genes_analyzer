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
from src.analysis.module_extractor import ModuleExtractor
from src.evolution.selector import SignatureGeneticSelector
from src.evaluation.reporter import PipelineReporter
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import ResultVisualizer

def main(config_path: str = "config.yaml"):
    config = Config(config_path)
    paths = PathManager(config.config)
    paths.ensure_dirs()

    logger = setup_logger(name="CancerAnalyzer", log_file=str(paths.log_file), level=logging.INFO)
    logger.info("Starting pancancer analysis pipeline")

    seed = config.config.get('global_seed', 229)
    RandomStateManager(seed=seed)

    logger.info("Stage 1 data aggregation, normalization")
    preprocessor = DataPreprocessor(config.config)
    X_raw, y, gene_names, merged_df = preprocessor.load_and_process()

    if X_raw.size == 0:
        logger.error("Pipeline halted, no data loaded")
        return

    logger.info("Stage 2 driver vs passenger classification")
    driver_classifier = DriverClassifier(config.config)
    X_drivers, driver_names, driver_stats = driver_classifier.filter_drivers(X_raw, gene_names)
    driver_stats.to_csv(paths.reports_dir / "driver_statistics.csv", index=False)

    logger.info("Stage 3 ASW-NMF-SS module extraction")
    extractor = ModuleExtractor(config.config)
    W, H, stable_genes, info = extractor.extract_modules(X_drivers, driver_names)

    optimal_k = info['optimal_k']
    logger.info(f"Extracted {optimal_k} modules")
    logger.info(f"Stable genes {len(stable_genes)}")
    
    module_genes = info['module_genes']
    for i, genes in enumerate(module_genes):
        logger.info(f"Module {i} top genes {', '.join(genes)}")

    logger.info("Stage 4 preparing features for GA")
    X_ga = W
    ga_feature_names = [f"Module_{i}" for i in range(optimal_k)]
    logger.info(f"GA will optimize over {optimal_k} module activation scores")

    logger.info("Stage 5 coevolutionary GA")
    ga = SignatureGeneticSelector(config.config, X_ga, y, ga_feature_names)
    best_chromosome, best_fitness, best_features = ga.run(verbose=True)

    logger.info("Stage 6 reporting and Evaluation")
    selected_indices = np.where(best_chromosome == 1)[0]

    evaluator = ModelEvaluator(config.config, paths)
    eval_results = evaluator.evaluate_and_save(X_ga, y, selected_indices, ga_feature_names)

    visualizer = ResultVisualizer(paths)
    visualizer.plot_evolution(ga.history)
    visualizer.plot_feature_importance(eval_results['importance_df'])

    unique_classes = sorted(np.unique(y))
    class_names = [f"Class_{int(c)}" for c in unique_classes]
    visualizer.plot_confusion_matrix(eval_results['y_test'], eval_results['y_pred'])

    reporter = PipelineReporter(paths)
    reporter.generate_final_report(
        y_test=eval_results['y_test'],
        y_pred=eval_results['y_pred'],
        y_proba=eval_results['y_proba'],
        cluster_names=class_names,
        best_signatures=best_features,
        signature_importance=eval_results['importance_df']
    )
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()