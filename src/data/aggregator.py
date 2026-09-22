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
        logger.info(f"DataAggregator initialized with {len(self.sources)} sources")

    def load_all(self) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        try:
            all_dfs = []
            all_gene_sets = []

            for i, source in enumerate(self.sources):
                logger.info(f"Loading source {i+1}/{len(self.sources)}: {source['path']}")
                df = self._load_single_source(source)

                if df is None or df.empty:
                    logger.warning(f"Source {source['path']} is empty, skipping")
                    continue

                logger.info(f"Extracting gene columns from source {i+1}")
                gene_cols = self._extract_gene_columns(df, source)
                logger.info(f"Found {len(gene_cols)} gene columns")

                logger.info(f"Binarizing mutations for source {i+1}")
                df = self._binarize_and_track(df, gene_cols, source)
                logger.info(f"Binarization complete for source {i+1}")

                unified_df = df[gene_cols].copy()
                unified_df['source_id'] = i

                all_dfs.append(unified_df)
                all_gene_sets.append(set(gene_cols))
                logger.info(f"Source {i+1} processed {len(gene_cols)} genes, {len(df)} samples")

            if not all_dfs:
                logger.error("No valid data sources loaded")
                return np.array([]), [], pd.DataFrame()

            logger.info("Merging datasets")
            merged_df = self._merge_datasets(all_dfs, all_gene_sets)
            logger.info(f"Merged dataset shape {merged_df.shape}")

            self.coverage_tracker.log_summary()

            high_conf_genes = self.coverage_tracker.get_high_confidence_genes()
            logger.info(f"High confidence genes {len(high_conf_genes)}")

            final_cols = ['source_id'] + sorted(high_conf_genes)
            final_cols = [c for c in final_cols if c in merged_df.columns]

            merged_df = merged_df[final_cols]
            gene_cols = sorted(high_conf_genes)

            X = merged_df[gene_cols].values

            logger.info(f"Aggregation complete {len(gene_cols)} genes, {len(merged_df)} samples")
            logger.info(f"Final dataset shape X={X.shape}")

            return X, gene_cols, merged_df

        except Exception as e:
            logger.error(f"Error in load_all: {e}", exc_info=True)
            raise

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
            logger.error(f"Failed to read {path} {e}", exc_info=True)
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
        indicator = source.get('mutation_indicator')

        for col in gene_cols:
            try:
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
            except Exception as e:
                logger.error(f"Error processing column {col} {e}", exc_info=True)
                raise

        return df

    def _merge_datasets(self, dfs: List[pd.DataFrame], gene_sets: List[set]) -> pd.DataFrame:
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