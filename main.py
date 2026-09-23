import warnings
import logging
import numpy as np
import pandas as pd
import json

warnings.filterwarnings("ignore")

from src.utils.logger import setup_logger
from src.utils.paths import PathManager
from src.utils.config_loader import Config
from src.utils.random_state import RandomStateManager
from src.data.preprocessor import DataPreprocessor
from src.analysis.driver_classifier import DriverClassifier
from src.analysis.module_extractor import ModuleExtractor
from src.evolution.selector import UnsupervisedModuleSelector
from src.evaluation.visualizer import ResultVisualizer
from src.evaluation.reporter import UnsupervisedReporter
from src.analysis.validator import ModuleValidator


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
    
def main(config_path: str = "config.yaml"):
    config = Config(config_path)
    paths = PathManager(config.config)
    paths.ensure_dirs()

    logger = setup_logger(name="CancerAnalyzer", log_file=str(paths.log_file), level=logging.INFO)

    logger.info("Starting pancancer analysis pipeline")

    RandomStateManager(seed=config.config.get('global_seed', 229))

    logger.info("Stage 1 data aggregation, normalization")
    preprocessor = DataPreprocessor(config.config)
    X, gene_names, merged_df = preprocessor.load_and_process()
    if X.size == 0:
        logger.error("Pipeline halted, no data loaded")
        return

    logger.info("Stage 2 driver vs passenger classification")
    driver_classifier = DriverClassifier(config.config)
    X_drivers, driver_names, driver_stats = driver_classifier.filter_drivers(X, gene_names)
    driver_stats.to_csv(paths.reports_dir / "driver_statistics.csv", index=False)

    logger.info("Stage 3 ASW-NMF-SS module extraction")
    extractor = ModuleExtractor(config.config)
    W, H, stable_genes, info = extractor.extract_modules(X_drivers, driver_names)

    optimal_k = info['optimal_k']
    module_genes = info['module_genes']
    module_names = [f"Module_{i}" for i in range(optimal_k)]

    logger.info(f"Extracted {optimal_k} modules, {len(stable_genes)} stable genes")
    for i, genes in enumerate(module_genes):
        logger.info(f"  {module_names[i]}: {', '.join(genes)}")

    logger.info("Stage 3.5 Unsupervised module validation")
    gene_to_idx_map = {g: i for i, g in enumerate(gene_names)}
    validator = ModuleValidator(
        V_binary=X,
        gene_names=gene_names,
        gene_to_idx=gene_to_idx_map
    )

    validation_report = {}
    for i, genes in enumerate(module_genes):
        module_name = f"Module_{i}"
        val_result = validator.validate_module(module_name, genes)
        validation_report[module_name] = val_result

        me = val_result['mutual_exclusivity']
        logger.info(
            f"{module_name}: {me['significant_pairs']}/{me['total_pairs_tested']}"
            f"exclusive gene pairs, mean or {me['mean_odds_ratio']:.2f}"
        )

        if val_result['pathway_enrichment']:
            top_pw = val_result['pathway_enrichment'][0]['pathway']
            logger.info(f"Enriched in {top_pw}")

    logger.info("Stage 4 unsupervised GA module selection")
    ga = UnsupervisedModuleSelector(config.config, W, module_names, V_binary=X_drivers, H=H, gene_names=driver_names)
    best_chrom, best_fitness, best_modules = ga.run(verbose=True)

    logger.info("Stage 5 reporting and visualization")
    visualizer = ResultVisualizer(paths)
    visualizer.plot_evolution(ga.history)

    importance_df = pd.DataFrame({
            'Feature': module_names,
            'Importance': np.var(W, axis=0)
    }).sort_values(by='Importance', ascending=False)
    visualizer.plot_feature_importance(importance_df)

    if ga.history['me_scores']:
        logger.info(
            f"Final fitness components"
            f"Variance {ga.history['variance_scores'][-1]:.3f}, "
            f"Redundancy {ga.history['redundancy_scores'][-1]:.3f}, "
            f"Mutual Exclusivity {ga.history['me_scores'][-1]:.3f}, "
            f"Manifold Separation {ga.history['manifold_scores'][-1]:.3f}"
            )
    reporter = UnsupervisedReporter(paths)
    reporter.generate_module_report(W, H, module_names, module_genes, stable_genes, driver_names)

    validation_path = paths.reports_dir / "validation_report.json"
    with open(validation_path, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    logger.info(f"Validation report saved to {validation_path}")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()