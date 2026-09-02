import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from .aggregator import DataAggregator

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.aggregator = DataAggregator(config.get('data_sources', []))
        logger.info("Data preproccesor ready")

    def load_and_process(self, filepath: str) -> tuple:
        logger.info("Starting data aggregation and preproccessing")

        X, y, gene_cols, merged_df = self.aggregator.load_all()

        if X.size == 0:
            logger.error("No data loaded")
            return np.array([]), np.array([]), [], pd.DataFrame()

        valid_mask = y >= 0
        if not valid_mask.all():
            invalid_count = (~valid_mask).sum()
            logger.warning(f"Filtered out {invalid_count} samples with invalid target")
            X = X[valid_mask]
            y = y[valid_mask]
            merged_df = merged_df[valid_mask].reset_index(drop=True)

        logger.info(f"Final dataset {X.shape[0]} samples, {X.shape[1]} genes")
        return X,y,gene_cols,merged_df