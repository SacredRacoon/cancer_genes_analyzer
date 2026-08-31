import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from ..models import Model

def plot_evolution(history,save_path = "output/fitness_history.png"):
    fig,ax = plt.subplots(figsize=(12,6))

    generations = range(len(history['best_fitness']))
    ax.plot(generations, history['best_fitness'], label='Лучшая особь', color='blue')
    ax.plot(generations, history['mean_fitness'], label='Средняя приспособленность', color='green')
    ax.fill_between(generations, np.array(history['best_fitness']), np.array(history['mean_fitness']), color='lightblue', alpha=0.2)
    ax.set_xlabel('Поколение')
    ax.set_ylabel('Валидация')
    ax.set_title('Эволюция га отбора генов')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path,dpi=150)
    plt.show()

def evaluate_model(x, y, selected_indices, feature_names, 
                   model_config=None,
                   save_model_path="output/models/best_model.pkl",
                   features_save_path="output/reports/selected_features.json",
                   feature_importance_path="output/plots/feature_importance.png",
                   confusion_matrix_path="output/plots/confusion_matrix.png"):

    x_selected = x[:, selected_indices]
    selected_names = [feature_names[i] for i in selected_indices]

    x_train, x_test, y_train, y_test = train_test_split(
        x_selected, y, test_size=0.2, random_state=42
    )
    
    if model_config is None:
        model_config = {
            'type': 'random_forest',
            'random_state': 42,
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 5
            }
        }
    
    models = Model()
    model = models.create_model(model_config)
    model.fit(x_train, y_train)

    Path(save_model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_model_path)
    print(f"Модель {save_model_path}")

    with open(features_save_path, "w") as f:
        json.dump(selected_names, f, indent=2)
    print(f"Список генов {features_save_path}")


    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1] if len(np.unique(y)) > 1 else None

    print(f"Выбранные признаки ({len(selected_names)}): {', '.join(selected_names)}")
    print(classification_report(y_test, y_pred, 
                               target_names=['LGG', 'GBM'] if len(np.unique(y)) > 1 else ['class 0']))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC {auc:.4f}")

    importance_df = pd.DataFrame({
        'Ген': selected_names,
        'Важность': model.feature_importances_
    }).sort_values(by='Важность', ascending=False)

    print(f"Важность признаков")
    print(importance_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Важность', y='Ген', data=importance_df, ax=ax)
    ax.set_title('Важность признаков модели')
    ax.set_xlabel('Важность')
    plt.tight_layout()
    plt.savefig(feature_importance_path, dpi=150, bbox_inches='tight')
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['LGG', 'GBM'] if len(np.unique(y)) > 1 else ['Class 0'],
                yticklabels=['LGG', 'GBM'] if len(np.unique(y)) > 1 else ['Class 0'])
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.set_title('Матрица ошибок')
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, importance_df

def load_model_and_predict(patient_mutations, 
                           model_path="output/best_model.pkl",
                           features_path="output/selected_features.json"):
    import json
    model = joblib.load(model_path) 
    
    with open(features_path, "r") as f:
        selected_features = json.load(f)
    
    print(f"Ожидаемые гены {selected_features}")

    x_new = np.zeros((1, len(selected_features)))
    for i, gene in enumerate(selected_features):
        if gene in patient_mutations:
            x_new[0, i] = 1 if patient_mutations[gene] == 'MUTATED' else 0

    proba = model.predict_proba(x_new)[0]
    
    if len(proba) > 1:
        return {f'LGG {proba[0]}, GBM {proba[1]} \nПредсказание': 'GBM' if proba[1] > 0.5 else 'LGG'}
    return proba

def prob_mutation_impact(model,patient_mutations,feature_names):
    x_new = np.zeros((1, len(feature_names)))
    for i, gene in enumerate(feature_names):
        if gene in patient_mutations:
            x_new[0, i] = 1 if patient_mutations[gene] == 'MUTATED' else 0

    proba = model.predict_proba(x_new)[0]
    if len(proba) > 1:
        return {f'LGG {proba[0]}, GBM {proba[1]} \nПредсказание': 'GBM' if proba[1] > 0.5 else 'LGG'}
    return proba