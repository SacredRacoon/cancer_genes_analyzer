import logging
from typing import Dict

logger = logging.getLogger(__name__)

class CoverageTracker:
    def __init__(self, min_tested_threshold: int = 50):
        self.min_tested_threshold = min_tested_threshold
        self.gene_stats: Dict[str, Dict[str, int]] = {}

    def record_gene_status(self, gene_name: str, is_tested: bool, is_mutated: bool):
        if gene_name not in self.gene_stats:
            self.gene_stats[gene_name] = {'tested': 0, 'mutated': 0}
        if is_tested:
            self.gene_stats[gene_name]['tested'] += 1
            if is_mutated:
                self.gene_stats[gene_name]['mutated'] +=1

    def add_bulk_stats(self, gene_name: str, tested_count: int, mutated_count: int):
        if gene_name not in self.gene_stats:
            self.gene_stats[gene_name] = {'tested': 0, 'mutated': 0}
        self.gene_stats[gene_name]['tested'] += int(tested_count)
        self.gene_stats[gene_name]['mutated'] += int(mutated_count)

    def get_true_frequency(self, gene_name: str) -> float:
        stats = self.gene_stats.get(gene_name, {'tested': 0, 'mutated': 0})
        if stats['tested'] == 0:
            return 0.0
        return stats['mutated'] / stats['tested']

    def get_tested_count(self, gene_name: str) -> int:
        return self.gene_stats.get(gene_name, {'tested': 0})['tested']

    def get_high_confidence_genes(self) -> list:
        return [
            gene for gene, stats in self.gene_stats.items()
            if stats['tested'] >= self.min_tested_threshold
        ]

    def log_summary(self):
        total_genes = len(self.gene_stats)
        high_conf_genes = len(self.get_high_confidence_genes())
        logger.info(f'Coverage tracker sum {total_genes} total genes'
                    f"{high_conf_genes} high confidence genes (tested .= {self.min_tested_threshold})")