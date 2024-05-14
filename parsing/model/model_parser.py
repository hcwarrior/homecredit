from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import tensorflow.keras as tf_keras
import yaml

from model.catboost_ import CatBoost
from model.lightgbm_ import LightGBM
from model.xgboost_ import XGBoost


@dataclass
class ModelConf:
    model_name: str
    model: str
    features: List[str]
    train_root_dir: str


@dataclass
class Model:
    model: Any
    conf: ModelConf


class ModelParser:
    def __init__(self,
                 model_yaml_path: str,
                 feature_conf: Dict[str, tf_keras.layers.Layer]):
        self.model_yaml_path = model_yaml_path
        self.feature_conf = feature_conf

    def parse(self) -> List[Model]:
        model_confs = self._parse_model_conf()
        result = []
        for model_conf in model_confs:
            feature_conf = {col: transformation for col, transformation in self.feature_conf.items()
                             if col in model_conf.features + ['target']}

            if model_conf.model == 'xgboost':
                model = XGBoost(feature_conf)
            elif model_conf.model == 'lightgbm':
                model = LightGBM(feature_conf)
            else:  # elif model_conf.model == 'catboost':
                model = CatBoost(feature_conf)

            result.append(Model(model=model, conf=model_conf))

        return result

    def _parse_model_conf(self) -> List[ModelConf]:
        with open(self.model_yaml_path) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

            result = []
            if 'features' not in conf:
                raise Exception('Please define "features".')

            features = conf['features']
            models = conf['models']
            for model_name, info in models.items():
                if 'train_root_dir' not in info:
                    raise Exception('Please define "train_root_dir".')

                result.append(ModelConf(model_name, info['type'], features, info['train_root_dir']))
            return result
