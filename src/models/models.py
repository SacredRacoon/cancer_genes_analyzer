from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np

class Model:
    """
    Юзабельные пока 4
    - random_forest: RandomForestClassifier
    - xgboost: XGBClassifier
    - lightgbm: LGBMClassifier
    - gradient_boosting: GradientBoostingClassifier
    Для добавления новых изменить параметры в конфиге, добавив новую модель
    А также здесь код для нее
    """

    def __init__(self):
        self.model_map = {
            'random_forest': RandomForestClassifier,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier,
            'gradient_boosting': GradientBoostingClassifier}
        
    def get_available_models(self):
        return list(self.model_map.keys())

    def create_model(self,model_config):
        model_type = model_config.get('type', 'random_forest')
        if model_type not in self.model_map:
            available = ', '.join(self.model_map.keys())
            raise ValueError(
                f"модели нет в списке, жуй какашки {model_type} "
            )

        model_class = self.model_map[model_type]
        params = self._prepare_params(model_type, model_config)
        model = model_class(**params)

        #print(f"модель {model_type} параметры {params}")
        return model

    def _prepare_params(self,model_type,model_config):
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