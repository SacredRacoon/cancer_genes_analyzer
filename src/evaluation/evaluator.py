import numpy as np
import pandas as pd
import json
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from ..models.classifiers import ModelFactory

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config: dict, path_manager):
        self.config = config
        self.paths = path_manager
        self.model_factory = ModelFactory()

    def evaluate_and_save(self, x: np.ndarray, y: np.ndarray, selected_indices: list, feature_name: list) -> dict:
        x_selected = x[:, selected_indices]
        selected_names = [feature_name[i] for i in selected_indices]

        x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size= 0.2, random_state=42)

        model = self.model_factory.create_model(self.config.get('model',{}))
        model.fit(x_train, y_train)

        model_path = self.paths.models_dir / "best_model.pkl"
        joblib.dump(model,model_path)
        logger.info(f"Model saved {model_path}")

        feature_path = self.paths.reports_dir / "selected_features.json"
        with open(feature_path, "w", encoding='utf-8') as f:
            json.dump(selected_names, f, indent=2)
        logger.info(f"Selected features in {feature_path}")

        y_pred = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:,1] if len(np.unique(y)) > 1 else None

        logger.info(f"Classification \n{classification_report(y_test, y_pred, target_names=['LGG','GBM'])}")

        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        logger.info(f"ROC AUC {auc:.4f}")

        importance_df = pd.DataFrame({
            'Feature': selected_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        return {
            'model': model,
            'importance_df': importance_df,
            'y_test': y_test,
            'y_pred': y_pred
        }

    def predict_patient(self, patient_data: dict, model, selected_features: list, feature_names: list) -> dict:
        x_new = np.zeros((1, len(selected_features)))
        for i, gene in enumerate(selected_features):
            if gene in patient_data:
                val = str(patient_data[gene]).upper()
                x_new[0,i] = 1 if val == 'MUTATED' else 0
            elif gene in ['age_years', 'sex']:
                x_new[0,i] = float(patient_data.get(gene, 0))

        proba = model.predict_proba(x_new)[0]
        prediction = 'GBM' if proba[1] > 0.5 else 'LGG'

        return {
            'prediction': prediction,
            'probabilities': {'LGG': round(proba[0], 4), 'GBM': round(proba[1],4)}
        }
