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
        result = {col: array_dict[col] for col in data_parser.features}, array_dict[target]
        if id is not None:
            result = result + array_dict[id]
        yield result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")

    args = parser.parse_args()
    options = args.options

    feature_conf = _parse_features(options.feature_yaml_path)
    model = _parse_model(options.model_yaml_path, feature_conf)
    keras_model, model_conf = model.model, model.conf

    # TODO: Please add optimizer as a parameter
    keras_model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.01),
                        loss='binary_crossentropy', metrics=[tf_keras.metrics.AUC()])

    print('Fitting a model...')
    train_data_parser = DatasetGenerator(options.train_data_root_dir, model_conf.features, model_conf.target, model_conf.id)
    train_data_generator = _generate_datasets(train_data_parser, model_conf.target)

    val_data_parser = DatasetGenerator(options.train_data_root_dir, model_conf.features, model_conf.target, model_conf.id)
    validation_data_generator = _generate_datasets(val_data_parser, model_conf.target)


    fd = tempfile.TemporaryFile('w+t')
    model_checkpoint_callback = tf_keras.callbacks.ModelCheckpoint(
        filepath=fd.name,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    keras_model.fit(x=train_data_generator, validation_data=validation_data_generator, callbacks=[model_checkpoint_callback],
                    steps_per_epoch=len(train_data_parser.files),
                    validation_steps=len(val_data_parser.files),
                    epochs=10,
                    class_weight={0: 0.51622883, 1: 15.904686})

    fd.close()

    # load the best model
    keras_model.load_weights(fd.name)

    os.remove(fd.name)

    keras_model.save(options.best_model_output_path)

    test_data_generator = _generate_datasets(options.test_data_root_dir, model_conf.features, model_conf.target)
    eval_df = pd.DataFrame({'case_id': [], 'target': [], 'score': []})
    for test_data_dict, target, case_id in test_data_generator:
        preds = keras_model.predict(test_data_dict).reshape((-1,))
        eval_df = pd.concat([eval_df,
                             pd.DataFrame({'case_id': case_id, 'target': target, 'score': preds})],
                            axis=0,
                            ignore_index=True)

    eval_df[['case_id', 'target']] = eval_df[['case_id', 'target']].astype(int)
    log_loss = sklearn.metrics.log_loss(eval_df['target'], eval_df['score'])
    auroc = sklearn.metrics.roc_auc_score(eval_df['target'], eval_df['score'])
    print(f'Log-loss / AUROC from test data set: {log_loss} / {auroc}')

    print('Saving results to the submission CSV file...')
    eval_df.drop(columns=['target']).to_csv(options.submission_csv_file_path, index=False)

