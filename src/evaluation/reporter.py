import pandas as pd
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class UnsupervisedReporter:
    def __init__(self, path_manager):
        self.paths = path_manager

    def generate_module_report(self, W, H, module_names, module_genes, stable_genes, driver_names):
        logger.info("Generating unsupervised module report")

        report_path = self.paths.reports_dir / "module_report.json"
        n_patients = W.shape[0]

        modules_info = {}
        for i, name in enumerate(module_names):
            # Статистика активаций
            activations = W[:, i]
            high_activation = np.sum(activations > np.percentile(activations, 90))

            modules_info[name] = {
                'top_genes': module_genes[i],
                'n_genes': len(module_genes[i]),
                'mean_activation': round(float(np.mean(activations)), 4),
                'max_activation': round(float(np.max(activations)), 4),
                'std_activation': round(float(np.std(activations)), 4),
                'patients_top10pct': int(high_activation),
                'patients_top10pct_pct': round(high_activation / n_patients * 100, 2)
            }

            logger.info(f"  {name}: mean={np.mean(activations):.3f}, "
                       f"top10%={high_activation} patients ({high_activation/n_patients*100:.1f}%)")
            logger.info(f"    Genes: {', '.join(module_genes[i])}")

        report = {
            'n_modules': len(module_names),
            'n_stable_genes': len(stable_genes),
            'n_total_drivers': len(driver_names),
            'n_patients': n_patients,
            'modules': modules_info,
            'stable_genes': stable_genes
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Module report saved to {report_path}")
        logger.info("MODULE SUMMARY")
        for name, info in modules_info.items():
            logger.info(f"  {name} ({info['patients_top10pct_pct']}% active): {', '.join(info['top_genes'][:4])}...")
        logger.info("="*60)