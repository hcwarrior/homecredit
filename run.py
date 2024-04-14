from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
import sklearn
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
    train_data_root_dir: str  # A root directory that training data files exist
    val_data_root_dir: str  # A root directory that validation data files exist
    test_data_root_dir: str  # A root directory that test data files exist
    submission_csv_file_path: str # A submission CSV output file path


def _parse_features(feature_yaml_path: str) -> Dict[str, BaseTransformation]:
    feature_parser = FeatureParser()
    feature_parser.load_prop(feature_yaml_path)

    return feature_parser.conf


def _parse_model(model_yaml_path: str, feature_conf: Dict[str, BaseTransformation]) -> Model:
    model_parser = ModelParser(model_yaml_path, feature_conf)

    return model_parser.parse()


def _generate_datasets(
        data_root_dir: str, input_cols: List[str], target: str) -> Iterator[Dict[str, np.ndarray]]:
    data_parser = DatasetGenerator(data_root_dir, input_cols, target)

    for file_path, array_dict in data_parser.parse():
        print(f'\nParsing {file_path}...')
        yield {col: array_dict[col] for col in input_cols}, array_dict[target]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")

    args = parser.parse_args()
    options = args.options

    feature_conf = _parse_features(options.feature_yaml_path)
    model = _parse_model(options.model_yaml_path, feature_conf)
    keras_model, model_conf = model.model, model.conf

    # TODO: Please add optimizer as a parameter
    keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf_keras.metrics.AUC()])

    print('Fitting a model...')
    train_data_generator = _generate_datasets(options.train_data_root_dir, model_conf.features, model_conf.target)
    validation_data_generator = _generate_datasets(options.val_data_root_dir, model_conf.features, model_conf.target)
    keras_model.fit(train_data_generator, validation_data=validation_data_generator)

    test_data_generator = _generate_datasets(options.test_data_root_dir, model_conf.features, model_conf.target)
    preds = []
    eval_df = pd.DataFrame({'case_id': [], 'target': [], 'score': []})
    for test_data in test_data_generator:
        eval_df = pd.concat([eval_df,
                             {'case_id': test_data['case_id'], 'target': test_data['target'], 'score': test_data['score']}],
                            ignore_index=True)

    log_loss = sklearn.metrics.log_loss(eval_df['target'], eval_df['score'])
    auroc = sklearn.metrics.roc_auc_score(eval_df['target'], eval_df['score'])
    print(f'Log-loss / AUROC from test data set: {log_loss} / {auroc}')

    print('Saving results to the submission CSV file...')
    eval_df.drop(columns=['target']).to_csv(options.submission_csv_file_path, index=False)
