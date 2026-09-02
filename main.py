import warnings
import logging
import numpy as np
warnings.filterwarnings("ignore")

from src.utils.logger import setup_logger
from src.utils.paths import PathManager
from src.utils.config_loader import Config
from src.data.preprocessor import DataPreprocessor
from src.genetic.selector import GeneticSelector
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import ResultVisualizer

def main(config_path: str = "config.yaml"):
    config = Config(config_path)
    paths = PathManager(config.config)
    paths.ensure_dirs()

    logger = setup_logger(name="GA_Glioma", log_file=str(paths.log_file), level=logging.INFO)
    logger.info("Starting GA glioma analysis pipeline")

    logger.info("Loading and preprocesssing data")
    preprocessor = DataPreprocessor(config.config)
<<<<<<< HEAD
    X, y, feature_names, df = preprocessor.load_and_process(str(paths.raw_data))

    if X.size == 0:
=======
    x, y, feature_names, df = preprocessor.load_and_process(str(paths.raw_data))

    if x.size == 0:
>>>>>>> 2aa54ff958acfe89cbbfc1c9b43090e57b3817b0
        logger.error("Data loading failed")
        return

    logger.info("Init genetic selector")
<<<<<<< HEAD
    ga = GeneticSelector(x=X, y=y, feature_names=feature_names, config=config.config)
=======
    ga = GeneticSelector(x=x, y=y, feature_names=feature_names, config=config.config)
>>>>>>> 2aa54ff958acfe89cbbfc1c9b43090e57b3817b0
    best_chromosome, best_fitness, best_features = ga.run(verbose=True)

    logger.info("Plotting GA evolution")
    visualizer = ResultVisualizer(paths)
    visualizer.plot_evolution(ga.history)

    logger.info("Evaluating and saving final model")
    evaluator = ModelEvaluator(config.config, paths)
    selected_indices = np.where(best_chromosome == 1)[0]

<<<<<<< HEAD
    evel_results = evaluator.evaluate_and_save(X, y, selected_indices, feature_names)
=======
    evel_results = evaluator.evaluate_and_save(x, y, selected_indices, feature_names)
>>>>>>> 2aa54ff958acfe89cbbfc1c9b43090e57b3817b0

    visualizer.plot_feature_importance(evel_results['importance_df'])
    visualizer.plot_confusion_matrix(evel_results['y_test'], evel_results['y_pred'])

<<<<<<< HEAD
    logger.info('ANALYZING GENES PER CANCER TYPE')

    cancer_gene_analysis = evaluator.analyze_genes_per_cancer_type(
        X, y, feature_names, selected_indices
    )   

    logger.info("FINAL REPORT: Genes per cancer type")
    for cancer_type, importance_df in cancer_gene_analysis.items():
        top_3 = importance_df.head(3)['Gene'].tolist()
        logger.info(f"Cancer Type {cancer_type} {', '.join(top_3)}")

=======
>>>>>>> 2aa54ff958acfe89cbbfc1c9b43090e57b3817b0
    logger.info("Running prediction for test patient")
    test_patient = config.get_test_patient()
    prediction_result = evaluator.predict_patient(
        patient_data=test_patient,
        model=evel_results['model'],
        selected_features=best_features,
        feature_names=feature_names
    )

    logger.info(f"Test patient predict {prediction_result['prediction']}")
<<<<<<< HEAD
    logger.info(f"Probabilities {prediction_result['probabilities']}")
=======
    logger.info(f"Probabilities LGG={prediction_result['probabilities']['LGG']}, GBM={prediction_result['probabilities']['GBM']}")
>>>>>>> 2aa54ff958acfe89cbbfc1c9b43090e57b3817b0

    logger.info("Pipleine completed successfully")

if __name__ == "__main__":
    main()