from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow.keras as tf_keras
import yaml

from layers.transformation.base_transformation import BaseTransformation
from model.dcn import DeepCrossNetwork


@dataclass
class ModelConf:
    features: List[str]
    target: str


@dataclass
class Model:
    model: tf_keras.Model
    conf: ModelConf


class ModelParser:
    def __init__(self,
                 model_yaml_path: str,
                 feature_conf: Dict[str, BaseTransformation]):
        self.model_yaml_path = model_yaml_path
        self.feature_conf = feature_conf

    def parse(self) -> Model:
        model_conf = self._parse_model_conf()
        self.feature_conf = {col: transformation for col, transformation in self.feature_conf.items()
                             if col in model_conf.features + [model_conf.target]}
        return DeepCrossNetwork(self.feature_conf)

    def _parse_model_conf(self) -> ModelConf:
        with open(self.model_yaml_path) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

            if 'features' not in conf:
                raise Exception('Please define "features".')

            return ModelConf(conf['features'], conf['target'])
