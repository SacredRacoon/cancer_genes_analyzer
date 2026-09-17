import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

logger = logging.getLogger(__name__)

class PipelineReporter:
    def __init__(self, path_manager):
        self.paths = path_manager

    def generate_final_report(self, y_test, y_pred, y_proba, cluster_names: list,
                              best_signatures: list, signature_importance: pd.DataFrame):
        logger.info("Generating final report")

        report_path = self.paths.reports_dir / "final_analysis_report.json"

        metrics = {
            'classification_report': classification_report(y_test, y_pred, target_names=cluster_names, output_dict=True),
            'roc_auc_macro': roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro') if len(np.unique(y_test)) > 2 else roc_auc_score(y_test, y_proba[:,1])
        }

        rules = []
        for _, row in signature_importance.head(5).iterrows():
            rules.append({
                'Signature': row['Signature'],
                'Importance': round(row['Importance'], 4),
                'Interpretation': f"Presence of this combination specific for cluster patterns"
            })

        final_report ={
            'metrics': metrics,
            'top_diagnostic_signatures': rules,
            'total_signatures_evaluated': len(best_signatures)
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        logger.info(f"Final report saved to {report_path}")

        logger.info("final diagnostic signatures discovered")
        for rule in rules:
            logger.info(f"{rule['Signature']}, importance {rule['Importance']}")