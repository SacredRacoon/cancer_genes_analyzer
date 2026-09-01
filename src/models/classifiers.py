from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import logging
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)

class ModelFactory:
    def __init__(self):
        self.model_map = {
            'random_forest': RandomForestClassifier,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier,
            'gradient_boosting': GradientBoostingClassifier}
        logger.info(f"Model factory ready, ready to go available {len(self.model_map)} models")

    def get_available_models(self):
        return list(self.model_map.keys())

    def create_model(self,model_config):
        model_type = model_config.get('type', 'random_forest').lower()

        if model_type not in self.model_map:
            available = ', '.join(self.model_map.keys())
            logger.error(f"Unsupported model {model_type}, available {available}")
            raise ValueError(f"Unsupported model type: '{model_type}'. Available: {available}")

        model_class = self.model_map[model_type]
        params = self._prepare_params(model_type, model_config)
        model = model_class(**params)

        logger.info(f"creating model {model_type} with params {params}")
        return model

    def _prepare_params(self,model_type: str,model_config: dict) -> dict:
        params = {'random_state': model_config.get('random_state', 42)}

        if model_type != 'xgboost' and 'n_jobs' in model_config:
            params['n_jobs'] = model_config['n_jobs']

        model_spec = model_config.get(model_type, {})
        params.update(model_spec)
        if model_type == 'xgboost':
            params.setdefault('verbosity', 0)
        elif model_type == 'lightgbm':
            params.setdefault('verbose', -1)

        return params