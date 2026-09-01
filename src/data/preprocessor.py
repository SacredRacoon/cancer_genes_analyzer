import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: dict):
        self.cfg = config.get('data_parsing', {})
        self.mutation_indicator = str(self.cfg.get('mutation_indicator', 'MUTATED')).upper()
        self.metadata_cols = [str(c).upper() for c in self.cfg.get('metadata_columns', [])]
        logger.info("Data preproccesor ready")

    def load_and_process(self, filepath: str) -> tuple:
        path = Path(filepath)
        if not path.exists():
            logger.error(f"Data file not foud {path}")
            return np.array([]), np.array([]), [], pd.DataFrame()

        try:
            logger.info(f"Loading data {path}")
            df = pd.read_csv(path)
            logger.info(f"Raw data shape {df.shape}")
        except Exception as e:
            logger.error(f"Failed reading csv {e}")
            return np.array([]), np.array([]), [], pd.DataFrame()

        df.columns = [str(c).strip() for c in df.columns]
        
        gene_cols = []
        for col in df.columns:
            if col.upper() not in self.metadata_cols:
                gene_cols.append(col)

        logger.info(f"Dynamically idf {len(gene_cols)} mutation/gene cols")

        for col in gene_cols:
            df[col] = (df[col].astype(str).str.upper() == self.mutation_indicator).astype(int)

        target_col = self.cfg.get('target_column', 'Grade')
        pos_class = str(self.cfg.get('target_positive_class', 'GBM')).upper()
        if target_col in df.columns:
            df['target'] = (df[target_col].astype(str).str.upper() == pos_class).astype(int)
        else:
            logger.error(f"Target column {target_col} not found in data")
            return np.array([]), np.array([]), df

        lgg_count = (df['target'] == 0).sum()
        gbm_count = (df['target'] == 1).sum()
        logger.info(f"Target distr LGG+{lgg_count}, GBM={gbm_count}")

        age_col = self.cfg.get('age_column', 'Age_at_diagnosis')
        if age_col in df.columns:
            df['age_years'] = df[age_col].astype(str).str.extract(r"(\d+)").astype(float).fillna(0.0)

        sex_col = self.cfg.get('sex_column', 'Gender')
        sex_pos = str(self.cfg.get('sex_positive_class', 'Male')).upper()
        if sex_col in df.columns:
            df['sex'] = (df[sex_col].astype(str).str.upper() == sex_pos).astype(int)

        feature_cols = gene_cols.copy()
        if 'age_years' in df.columns: feature_cols.append('age_years')
        if 'sex' in df.columns: feature_cols.append('sex')

        df[feature_cols] = df[feature_cols].fillna(0)

        x = df[feature_cols].values
        y = df['target'].values

        logger.info(f"Preprocessing complete, feature matrix {x.shape}")
        return x,y,feature_cols,df
