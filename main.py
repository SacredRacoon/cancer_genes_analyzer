import warnings
warnings.filterwarnings("ignore")

from src.utils.config_loader import Config
from src.data.loader import load_prep
from src.genetic.selector import Genetic_selector
from src.evaluation.evaluator import plot_evolution, evaluate_model, prob_mutation_impact
import numpy as np

def main(config_path="config.yaml"):
    print("конфиг грузится")
    config = Config(config_path)
    config.print_config()
    config.create_output_dirs()
    
    print("грузятся данные")
    x, y, feature_names, df = load_prep(
        config.get_paths()['raw_data'],
        genes=config.get_genes()
    )
    
    print("запускаем га")
    ga = Genetic_selector( 
        x=x,
        y=y,
        feature_name=feature_names,
        **config.get_ga_params(),
        model_config=config.get_model_params(),
        cv_params=config.get_cv_params()
    )
    
    best_chromosome, best_fitness, best_features = ga.run(verbose=True)

    print("визуализация")
    plot_evolution(ga.history, save_path=config.get_paths()['fitness_plot'])
    
    print("обучаем финальную модель")
    selected_indices = np.where(best_chromosome == 1)[0]
    model, importance_df = evaluate_model(
        x, y, selected_indices, feature_names,
        model_config=config.get_model_params(),
        save_model_path=config.get_paths()['model_save'],
        features_save_path=config.get_paths()['features_save'],
        feature_importance_path=config.get_paths()['importance_plot'],
        confusion_matrix_path=config.get_paths()['confusion_matrix_plot']
    )
    

    print(f"предсказание для тестового чувака")
    
    result = prob_mutation_impact(
        model,
        config.get_test_patient(),
        [feature_names[i] for i in selected_indices]
    )
    
    print("Результат предсказания")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print(f"Результаты в {config.get_paths()['output_dir']}")


if __name__ == "__main__":
    main()