from typing import Dict, Iterator

import numpy as np
import tensorflow.keras as tf_keras
from dataclasses import dataclass
from simple_parsing import ArgumentParser

from layers.transformation.base_transformation import BaseTransformation
from parsing.data.data_parser import DatasetGenerator
from parsing.feature.feature_parser import FeatureParser
from parsing.model.model_parser import ModelParser, ModelConf, Model


@dataclass
class Options:
    feature_yaml_path: str  # A feature YAML file path
    model_yaml_path: str  # A model YAML file path
    data_root_dir: str  # A root directory that data files exist


def _parse_features(feature_yaml_path: str) -> Dict[str, BaseTransformation]:
    feature_parser = FeatureParser()
    feature_parser.load_prop(feature_yaml_path)

    return feature_parser.conf


def _parse_model(model_yaml_path: str, feature_conf: Dict[str, BaseTransformation]) -> Model:
    model_parser = ModelParser(model_yaml_path, feature_conf)

    return model_parser.parse()


def _generate_datasets(data_root_dir: str) -> Iterator[Dict[str, np.ndarray]]:
    data_parser = DatasetGenerator(data_root_dir)

    for array_dict in data_parser.parse():
        yield array_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")

    args = parser.parse_args()
    options = args.options

    feature_conf = _parse_features(options.feature_yaml_path)
    model = _parse_model(options.model_yaml_path, feature_conf)
    keras_model, model_conf = model.model, model.conf

    print('Fitting a model...')
    for data in _generate_datasets(options.data_root_dir):
        keras_model.fit({col: data[col] for col in model_conf.features}, data[model_conf.target])

    print(keras_model.summary())