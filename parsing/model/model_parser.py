from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow.keras as tf_keras
import yaml

from model.dcn import DeepCrossNetwork
from model.xgboost_ import XGBoost


@dataclass
class ModelConf:
    model: str
    features: List[str]
    target: str
    id: str


@dataclass
class Model:
    model: tf_keras.Model
    conf: ModelConf


class ModelParser:
    def __init__(self,
                 model_yaml_path: str,
                 feature_conf: Dict[str, tf_keras.layers.Layer]):
        self.model_yaml_path = model_yaml_path
        self.feature_conf = feature_conf

    def parse(self) -> Model:
        model_conf = self._parse_model_conf()
        self.feature_conf = {col: transformation for col, transformation in self.feature_conf.items()
                             if col in model_conf.features + [model_conf.target]}

        model = XGBoost(self.feature_conf)
        return Model(model=model, conf=model_conf)

    def _parse_model_conf(self) -> ModelConf:
        with open(self.model_yaml_path) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

            if 'features' not in conf:
                raise Exception('Please define "features".')

            if 'label' not in conf:
                raise Exception('Please define "label" (y).')

            if 'id' not in conf:
                raise Exception('Please define "id".')

            return ModelConf(conf['model'], conf['features'], conf['label'], conf['id'])
