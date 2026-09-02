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

        target_names = [f'Cancer_{c}' for c in sorted(np.unique(y))]
        logger.info(f"Classification \n{classification_report(y_test, y_pred, target_names=target_names)}")

        if y_proba is not None and len(np.unique(y)) == 2:
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
            logger.info(f"ROC AUC {auc:.4f}")

        importance_df = pd.DataFrame({
            'Feature': selected_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        logger.info("Feature Importance calculated:\n" + importance_df.to_string(index=False))

        return {
            'model': model,
            'importance_df': importance_df,
            'y_test': y_test,
            'y_pred': y_pred
        }

    def analyze_genes_per_cancer_type(self, X:np.ndarray, y: np.ndarray,
                                      feature_names: list, selcted_indices: list) -> dict:
        logger.info("Analyzing per cancer type gene")

        cancer_types = sorted(np.unique(y))
        results = {}

        selected_names = [feature_names[i] for i in selcted_indices]
        X_selected = X[:, selcted_indices]

        for cancer_type in cancer_types:
            logger.info(f'Analyzing cancer type {cancer_type}')

            y_binary = (y == cancer_type).astype(int)

            if y_binary.sum() < 5:
                logger.warning(f"Not enough samples for cancer type {cancer_type}. Skipping")
                continue

            model = self.model_factory.create_model(self.config.get('model', {}))
            model.fit(X_selected, y_binary)

            importance_df = pd.DataFrame({
                'Gene': selected_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            results[cancer_type] = importance_df

            top_genes = importance_df.head(5)['Gene'].tolist()
            logger.info(f"Top genes for cancer type {cancer_type}: {top_genes}")

        report_path = self.paths.reports_dir / "per_cancer_type.json"
        report_data = {}
        for cancer_type, df in results.items():
            report_data[f"Cancer_{cancer_type}"] = df.head(10).to_dict(orient='records')

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Per cancer type gene importance saved to {report_path}")
        return results

    def predict_patient(self, patient_data: dict, model, selected_features: list, feature_names: list) -> dict:
        X_new = np.zeros((1, len(selected_features)))
        for i, gene in enumerate(selected_features):
            if gene in patient_data:
                val = str(patient_data[gene]).upper()
                X_new[0,i] = 1 if val == 'MUTATED' else 0

        proba = model.predict_proba(X_new)[0]

        if len(proba) >2:
            predicted_class = np.argmax(proba)
            probs_dict = {f"Cancer_{i}": round(p, 4) for i, p in enumerate(proba)}
            return {
                'prediction': f"Cancer_{predicted_class}",
                'probabilities': probs_dict
            }
        else:
            prediction = 'Cancer_1' if proba[1] > 0.5 else 'Cancer_0'
            return {
                'prediction': prediction,
                'probabilities': {'Cancer_0': round(proba[0], 4), 'Cancer_1': round(proba[1],4)}
            }
