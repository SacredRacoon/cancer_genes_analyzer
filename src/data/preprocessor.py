import pandas as pd
import numpy as np
import logging
from typing import Tuple, List
from .aggregator import DataAggregator

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        data_cfg = config.get('data_parsing', {})
        min_tested = data_cfg.get('min_tested_threshold', 50)
        self.aggregator = DataAggregator(
            data_sources=config.get('data_sources', []),
            min_tested_threshold=min_tested
        )
        logger.info("Data preprocessor ready")

    def load_and_process(self) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        logger.info("Starting data aggregation and preprocessing")
        X, gene_cols, merged_df = self.aggregator.load_all()

        if X.size == 0:
            logger.error("No data loaded")
            return np.array([]), [], pd.DataFrame()

        logger.info(f"Final dataset {X.shape[0]} samples, {X.shape[1]} genes")
        return X, gene_cols, merged_df