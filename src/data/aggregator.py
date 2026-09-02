import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class DataAggregator:
    def __init__(self, data_sources: List[Dict]):
        self.sources = data_sources
        logger.info(f"DataAggregator initialized with {len(self.sources)}")

    def load_all(self) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        all_dfs = []
        all_gene_sets = []

        for i, source in enumerate(self.sources):
            logger.info(f"Loading source {i+1}/{len(self.sources)}: {source['path']}")
            df = self._load_single_source(source)

            if df is None or df.empty:
                logger.warning(f"Source {source['path']} is empty, skipping")
                continue

            df = self._normalize_target_column(df, source)

            if 'target' not in df.columns:
                logger.warning(f"Skipping source {source['path']} due to target normalization failure")
                continue

            gene_cols = self._extract_gene_columns(df, source)
            df = self._binarize_mutations(df, gene_cols, source)

            unified_df = df[gene_cols].copy()
            unified_df['target'] = df['target']
            unified_df['source_id'] = i

            all_dfs.append(unified_df)
            all_gene_sets.append(set(gene_cols))
            logger.info(f"Source {i+1} processed {len(gene_cols)} genes, {len(df)} samples")

        if not all_dfs:
            logger.error("No valid data sources loaded")
            return np.array([]), np.array([]), [], pd.DataFrame()

        merged_df = self._merge_datasets(all_dfs, all_gene_sets)

        all_cols = merged_df.columns.tolist()
        gene_cols = [
            col for col in all_cols
            if col.lower() not in ['target','source_id']]
        gene_cols = sorted(list(set(gene_cols)))

        x = merged_df[gene_cols].values

        y_data = merged_df['target']
        if isinstance(y_data, pd.DataFrame):
            logger.warning("Duplicate 'target' columns detected! Taking the first one.")
            y_data = y_data.iloc[:, 0]
        y = y_data.astype(int).values.ravel()
        logger.info(f"Aggregation complete {len(gene_cols)} genes, {len(merged_df)} samples")
        logger.info(f"Final dataset shape X={x.shape}, y={y.shape}")
        logger.info(f"Unique cancer types in y {np.unique(y)}")

        return x, y, gene_cols, merged_df

    def _load_single_source(self, source: Dict) -> Optional[pd.DataFrame]:
        path = Path(source['path'])
        if not path.exists():
            logger.error(f"File {path} does not exist")
            return None

        fmt = source.get('format', 'csv').lower()
        try:
            if fmt == 'csv':
                df = pd.read_csv(path)
            elif fmt in ['tsv', 'txt']:
                df = pd.read_csv(path, sep='\t')
            else:
                logger.error(f"Unsupported file format {fmt}")
                return None

            logger.info(f"Loaded {len(df)} rows from {path}")
            return df
        except Exception as e:
            logger.error(f"Failed to read {path} {e}")
            return None

    def _extract_gene_columns(self, df: pd.DataFrame, source: Dict) -> List[str]:
        metadata = [str(c).upper() for c in source.get('metadata_columns', [])]
        target_col = source.get('target_column')

        gene_cols = []
        for col in df.columns:
            if col == target_col:
                continue
            if col.upper() in metadata:
                continue
            gene_cols.append(col)

        return gene_cols

    def _normalize_target_column(self, df: pd.DataFrame, source: Dict) -> pd.DataFrame:
        target_col = source.get('target_column')
        mapping = source.get('target_mapping')

        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found")
            return df

        if mapping:
            df['target'] = df[target_col].astype(str).map(mapping)
            if df['target'].isna().any():
                logger.warning(f"Some target values not in mapping, filling with -1")
                df['target'] = df['target'].fillna(-1).astype(int)

        else:
            df['target'] = pd.to_numeric(df[target_col], errors='coerce').fillna(-1).astype(int)

        return df

    def _binarize_mutations(self, df: pd.DataFrame, gene_cols: List[str], source: Dict) -> pd.DataFrame:
        indicator = source.get('mutation_indicator')

        for col in gene_cols:
            if indicator and df[col].dtype == object:
                df[col] = (df[col].astype(str).str.upper() == str(indicator).upper()).astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                df[col] = (df[col] == 1).astype(int)

        return df

    def _merge_datasets(self, dfs: List[pd.DataFrame], gene_sets: List[set]) -> pd.DataFrame:
        all_genes = set()
        for gs in gene_sets:
            all_genes.update(gs)
        all_genes = sorted(list(all_genes))

        logger.info(f"Total unique genes in all souces {len(all_genes)}")

        normalized_dfs = []
        for i, df in enumerate(dfs):
            missing_genes = set(all_genes) - set(df.columns)
            for gene in missing_genes:
                df[gene] = 0
            df = df[['target','source_id'] + all_genes]
            normalized_dfs.append(df)

        merged = pd.concat(normalized_dfs, ignore_index=True)
        logger.info(f"Merged dataset shape {merged.shape}")

        return merged
