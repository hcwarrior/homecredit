import os
import tempfile
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow.keras as tf_keras
from dataclasses import dataclass

from simple_parsing import ArgumentParser
from sklearn.utils import class_weight

from parsing.data.data_parser import DatasetGenerator
from parsing.feature.feature_parser import FeatureParser
from parsing.model.model_parser import ModelParser, ModelConf, Model


@dataclass
class Options:
    feature_yaml_path: str  # A feature YAML file path
    model_yaml_path: str  # A model YAML file path
    train_data_root_dir: str  # A root directory that training data files exist
    val_data_root_dir: str  # A root directory that validation data files exist
    test_data_root_dir: str  # A root directory that test data files exist
    submission_csv_file_path: str  # A submission CSV output file path
    best_model_output_path: str  # A path for the best model
    preprocess_json_path: str  # A path for preprocess json


def _parse_features(feature_yaml_path: str) -> Dict[str, tf_keras.layers.Layer]:
    feature_parser = FeatureParser()
    feature_parser.load_prop(feature_yaml_path)

    return feature_parser.conf


def _parse_model(model_yaml_path: str, feature_conf: Dict[str, object]) -> Model:
    model_parser = ModelParser(model_yaml_path, feature_conf)

    return model_parser.parse()


def _generate_datasets(data_parser: DatasetGenerator, target: str, id: str = None) -> Iterator[Dict[str, np.ndarray]]:

    for file_path, array_dict in data_parser.parse():
        print(f'\nParsing {file_path}...')
        # dict, target, id (optional)
        result = {col: array_dict[col] for col in data_parser.features if col != target}, array_dict[target]
        if id is not None:
            result = result + (array_dict[id], )
        yield result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")

    args = parser.parse_args()
    options = args.options

    feature_conf = _parse_features(options.feature_yaml_path)
    model = _parse_model(options.model_yaml_path, feature_conf)
    model, model_conf = model.model, model.conf

    print('Fitting a model...')
    train_data_parser = DatasetGenerator(options.train_data_root_dir, model_conf.features, model_conf.target, model_conf.id)
    train_data_generator = _generate_datasets(train_data_parser, model_conf.target)

    val_data_parser = DatasetGenerator(options.val_data_root_dir, model_conf.features, model_conf.target,
                                         model_conf.id)
    val_data_generator = _generate_datasets(val_data_parser, model_conf.target)

    val_df, val_target = None, None
    for val_data_dict, val_target_arr in val_data_generator:
        _val_tmp_df = pd.DataFrame.from_dict(val_data_dict)
        val_df = _val_tmp_df if val_df is None else pd.concat([val_df, _val_tmp_df], axis=0, ignore_index=True)
        val_target = val_target_arr if val_target is None else np.concatenate([val_target, val_target_arr], axis=0)

    for train_data_dict, target in train_data_generator:
        model.fit(pd.DataFrame.from_dict(train_data_dict), target, val_df, val_target)

    test_data_parser = DatasetGenerator(options.test_data_root_dir, model_conf.features, model_conf.target,
                                         model_conf.id)
    test_data_generator = _generate_datasets(test_data_parser, model_conf.target, model_conf.id)
    eval_df = pd.DataFrame({'case_id': [], 'target': [], 'score': []})

    test_df, test_target, test_case_id = None, None, None
    for test_data_dict, test_target_arr, case_id in test_data_generator:
        _test_tmp_df = pd.DataFrame.from_dict(test_data_dict)
        test_df = _test_tmp_df if test_df is None else pd.concat([test_df, _test_tmp_df], axis=0, ignore_index=True)
        test_target = test_target_arr if test_target is None else np.concatenate([test_target, test_target_arr], axis=0)
        test_case_id = case_id if test_case_id is None else np.concatenate([test_case_id, case_id], axis=0)

    preds, loss, auc = model.predict(test_df, test_target)
    print(f'Test AUC: {auc} / Log Loss: {loss}')
    eval_df = pd.concat([eval_df,
                         pd.DataFrame({'case_id': test_case_id, 'target': test_target, 'score': preds})],
                        axis=0,
                        ignore_index=True)

    print('Saving results to the submission CSV file...')
    eval_df.drop(columns=['target']).to_csv(options.submission_csv_file_path, index=False)

    print('Saving model...')
    model.save(options.best_model_output_path, options.preprocess_json_path)
