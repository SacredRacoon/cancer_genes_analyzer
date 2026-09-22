import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .coverage_tracker import CoverageTracker

logger = logging.getLogger(__name__)

class DataAggregator:
    def __init__(self, data_sources: List[Dict], min_tested_threshold: int = 50):
        self.sources = data_sources
        self.coverage_tracker = CoverageTracker(min_tested_threshold=min_tested_threshold)
        logger.info(f"DataAggregator initialized with {len(self.sources)}")

    def load_all(self) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        all_dfs = []
        all_gene_sets = []

        for i, source in enumerate(self.sources):
            logger.info(f"Loading source {i+1}/{len(self.sources)}: {source['path']}")
            df = self._load_single_source(source)

    def _load_single_source(self, source: Dict) -> Optional[pd.DataFrame]:
        path = Path(source['path'])
        if not path.exists():
            logger.error(f"File doesn't exist {path.resolve()}")
            return None

        fmt = source.get('format', 'csv').lower()
        try:
            if fmt == 'csv':
                df = pd.read_csv(path, encoding='utf-8')
            elif fmt in ['tsv', 'txt']:
                df = pd.read_csv(path, sep='\t', encoding='utf-8', on_bad_lines='skip')
            else:
                logger.error(f"Unsupported format {fmt}")
                return None
            logger.info(f"Loaded {len(df)} rows from {path.name}")
            return df
        except Exception as e:
            logger.error(f"Failed to read {path} {e}")
            return None

    def _extract_gene_columns(self, df: pd.DataFrame, source: Dict) -> List[str]:
        metadata = [str(c).strip().upper() for c in source.get('metadata_columns', [])]

        gene_cols = []
        for col in df.columns:
            col_clean = str(col).strip().upper()
            if col_clean in metadata:
                continue
            gene_cols.append(str(col).strip())
        return gene_cols

    def _binarize_and_track(self, df: pd.DataFrame, gene_cols: List[str], source: Dict) -> pd.DataFrame:
        indicator = self.source.get('mutation_indicator')

        for col in gene_cols:
            if indicator and df[col].dtype == object:
                is_mutated = (df[col].astype(str).str.upper() == str(indicator).upper())
                df[col] = is_mutated.astype(int)
                self.coverage_tracker.add_bulk_stats(col, len(df), int(is_mutated.sum()))
            else:
                original_is_na = df[col].isna()
                df[col] = pd.to_numeric(df[col], errors='coerce')
                tested = int((~original_is_na).sum())
                mutated = int((df[col] == 1).sum())
                self.coverage_tracker.add_bulk_stats(col, tested, mutated)
                df[col] = df[col].where(~original_is_na, np.nan)

        return df

    def _merge_dataset(self, dfs: List[pd.DataFrame], gene_sets: List[set]) -> pd.DataFrame:
        all_genes = set()
        for gs in gene_sets:
            all_genes.update(gs)
        all_genes = sorted(list(all_genes))

        logger.info(f"Total unique genes across all sources {len(all_genes)}")

        normalized_dfs = []
        for df in dfs:
            missing = set(all_genes) - set(df.columns)
            for gene in missing:
                df[gene] = np.nan
            df = df[['source_id'] + all_genes]
            normalized_dfs.append(df)

        merged = pd.concat(normalized_dfs, ignore_index=True)
        logger.info(f"Merged dataset shape {merged.shape}")
        return merged