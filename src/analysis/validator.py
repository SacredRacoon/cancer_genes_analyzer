import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
from scipy.stats import fisher_exact

logger = logging.getLogger(__name__)

class ModuleValidator:
    def __init__(self, V_binary: np.ndarray, gene_names: List[str], gene_to_idx: Dict[str, int]):
        self.V_binary = V_binary
        self.gene_names = gene_names
        self.gene_to_idx = gene_to_idx
        self.n_patient = V_binary.shape[0]

        try:
            import gseapy as gp
            self.gseapy_available = True
            logger.info("gseapy found")
        except ImportError:
            self.gseapy_available = False
            logger.warning("gseapy not found")

    def validate_module(self,module_name: str, top_genes: List[str]) -> Dict:
        logger.info(f"Validating {module_name} with {len(top_genes)} genes")

        me_results = self._test_mutual_exclusivity(top_genes)
        enrichment_results = self._run_enrichment(top_genes)

        return {
            'module': module_name,
            'genes': top_genes,
            'mutual_exclusivity': me_results,
            'pathway_enrichment': enrichment_results
        }

    def _test_mutual_exclusivity(self, genes: List[str]) -> Dict:
        valid_genes = [g for g in genes if g in self.gene_to_idx]
        if len(valid_genes) < 2:
            return {'significant_pairs': 0, 'mean_odds_ratio': 1.0, 'details': []}

        significant_pairs = 0
        odds_ratios = []
        details = []

        for i in range(len(valid_genes)):
            for j in range(i+1, len(valid_genes)):
                gene_a = valid_genes[i]
                gene_b = valid_genes[j]

                idx_a = self.gene_to_idx[gene_a]
                idx_b = self.gene_to_idx[gene_b]

                mut_a = self.V_binary[:, idx_a]
                mut_b = self.V_binary[:, idx_b]

                valid_mask = ~np.isnan(mut_a) & ~np.isnan(mut_b)

                if np.sum(valid_mask) < 10:
                    continue

                v_mut_a = mut_a[valid_mask]
                v_mut_b = mut_b[valid_mask]

                a = np.sum((v_mut_a == 1) & (v_mut_b == 1))
                b = np.sum((v_mut_a == 1) & (v_mut_b == 0))
                c = np.sum((v_mut_a == 0) & (v_mut_b == 1))
                d = np.sum((v_mut_a == 0) & (v_mut_b == 0))

                contingency_table = [[a,b], [c,d]]

                try:
                    oddsratio, p_value = fisher_exact(contingency_table, alternative='less')

                    a_corr, b_corr, c_corr, d_corr = a + 0.5, b +0.5, c +0.5, d + 0.5
                    or_corrected = (a_corr * d_corr) / (b_corr * c_corr)

                    or_corrected = min(or_corrected, 100.0)
                    log2_or = np.log2(or_corrected + 1)  
                    if p_value < 0.05:
                        significant_pairs += 1

                    odds_ratios.append(or_corrected)
                    details.append({
                        'gene_pair': f"{gene_a} & {gene_b}",
                        'tested_patients': int(np.sum(valid_mask)),
                        'co_occurrence': a,
                        'p_value': round(float(p_value), 4),
                        'odds_ratio': round(float(or_corrected), 4),
                        'log2_odds_ratio': round(float(log2_or), 2)
                    })
                except Exception:
                    continue

        mean_or = float(np.mean(odds_ratios)) if odds_ratios else 1.0

        return {
            'total_pairs_tested': len(details),
            'significant_pairs': significant_pairs,
            'mean_odds_ratio': round(mean_or, 4),
            'top_exclusive_pairs': sorted(details, key=lambda x: x['p_value'])[:3]
        }

    def _run_enrichment(self, genes: List[str]) -> List[Dict]:
        if not self.gseapy_available or len(genes) < 3:
            return []

        try:
            import gseapy as gp

            enr = gp.enrichr(
                gene_list=genes,
                gene_sets='KEGG_2021_Human',
                organism='human',
                outdir=None,
                cutoff=0.05,
                format='pdf',
                verbose=False
            )

            top_pathways = enr.results.head(3).to_dict(orient='records')

            cleaned_pathways = []
            for pw in top_pathways:
                cleaned_pathways.append({
                    'pathway': str(pw.get('Term', '')),
                    'p_value': float(pw.get('P-value', 1.0)),
                    'adjusted_p_value': float(pw.get('Adjusted P-value', 1.0)),
                    'genes_in_pathway': str(pw.get('Genes', '')),
                    'overlap': str(pw.get('Overlap', ''))
                })

            return cleaned_pathways
        
        except Exception as e:
            logger.warning(f"Enrichment analysis failed {e}")
            return []