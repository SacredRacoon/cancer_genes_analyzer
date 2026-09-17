import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

logger = logging.getLogger(__name__)

def safe_roc_auc(y_test, y_proba) -> float:
    if y_proba is None:
        return 0.0
        
    n_classes = len(np.unique(y_test))
    if n_classes < 2:
        logger.warning("Test set contains only 1 class")
        return 0.0

    try:
        if n_classes == 2:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                return roc_auc_score(y_test, y_proba[:, 1])
            elif y_proba.ndim == 1:
                return roc_auc_score(y_test, y_proba)
            else:
                return roc_auc_score(y_test, y_proba)
        else:
            return roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except Exception as e:
        logger.warning(f"Could not compute roc auc {e}")
        return 0.0


class PipelineReporter:
    def __init__(self, path_manager):
        self.paths = path_manager

    def generate_final_report(self, y_test, y_pred, y_proba, cluster_names: list,
                              best_signatures: list, signature_importance: pd.DataFrame):
        logger.info("Generating final report")

        report_path = self.paths.reports_dir / "final_analysis_report.json"

        auc_score = safe_roc_auc(y_test, y_proba)

        metrics = {
            'classification_report': classification_report(
                y_test, y_pred, 
                target_names=cluster_names, 
                output_dict=True, 
                zero_division=0 
            ),
            'roc_auc': round(auc_score, 4)
        }

        rules = []
        if not signature_importance.empty:
            for _, row in signature_importance.head(5).iterrows():
                rules.append({
                    'Signature': row.get('Feature', row.get('Signature', 'Unknown')), 
                    'Importance': round(float(row['Importance']), 4),
                    'Interpretation': "Presence of this combination is specific to cluster patterns"
                })

        final_report = {
            'metrics': metrics,
            'top_diagnostic_signatures': rules,
            'total_signatures_evaluated': len(best_signatures)
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        logger.info(f"Final report saved to {report_path}")
        
        logger.info("final diagnostic signatures")
        if rules:
            for rule in rules:
                logger.info(f"{rule['Signature']} (Importance {rule['Importance']})")
        else:
            logger.info("No specific diagnostic signatures passed GA filter.")