import os
import tempfile
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow.keras as tf_keras
from dataclasses import dataclass

from matplotlib import pyplot as plt
from simple_parsing import ArgumentParser
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.utils import class_weight

from parsing.data.data_parser import DatasetGenerator
from parsing.feature.feature_parser import FeatureParser
from parsing.model.model_parser import ModelParser, ModelConf, Model

pd.set_option('mode.chained_assignment',  None)

@dataclass
class Options:
    feature_yaml_path: str  # A feature YAML file path
    model_yaml_path: str  # A model YAML file path
    test_data_root_dir: str  # A root directory that test data files exist
    submission_csv_file_path: str  # A submission CSV output file path
    best_model_output_path: str  # A path for the best model
    preprocess_json_path: str  # A path for preprocess json
    output_auc_png_path: str  # output AUC png path (by WEEK_NUM)


def _parse_features(feature_yaml_path: str) -> Dict[str, tf_keras.layers.Layer]:
    feature_parser = FeatureParser()
    feature_parser.load_prop(feature_yaml_path)

    return feature_parser.conf


def _parse_model(model_yaml_path: str, feature_conf: Dict[str, object]) -> List[Model]:
    model_parser = ModelParser(model_yaml_path, feature_conf)

    return model_parser.parse()


def _generate_datasets(data_parser: DatasetGenerator, target: str, id: str, week_num: str, add_id: bool, add_weeknum: bool) -> Iterator[Dict[str, np.ndarray]]:
    for file_path, array_dict in data_parser.parse():
        print(f'\nParsing {file_path}...')
        # dict, target, id (optional)
        result = {col: array_dict[col] for col in data_parser.features if col not in (target, id, week_num)}, array_dict[target]
        if add_id:
            result = result + (array_dict[id], )
        if add_weeknum:
            result = result + (array_dict[week_num],)
        yield result


def _stability_metric(week, y_true, y_pred):
    """
    Custom metric for model optimization during training
    """
    weeks_to_score = week
    gini_in_time = []

    for week in weeks_to_score.unique():
        week_idx = weeks_to_score.eq(week)
        try:
            gini = np.array(2 * roc_auc_score(y_true[week_idx], y_pred[week_idx]) - 1)
            gini_in_time.append(gini)
        except Exception as e:
            continue

    w_fallingrate = 88.0
    w_resstd = -0.5
    x = np.arange(len(gini_in_time))
    y = np.array(gini_in_time)
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(y)
    stability_score = avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std
    is_higher_better = True

    return 'stability_score', stability_score, is_higher_better


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")

    args = parser.parse_args()
    options = args.options

    feature_conf = _parse_features(options.feature_yaml_path)
    models = _parse_model(options.model_yaml_path, feature_conf)

    print('Fitting a model...')
    result_df = None
    for model in models:
        model, model_conf = model.model, model.conf
        required_features = list(set(model_conf.features) | {'WEEK_NUM', 'case_id', 'target'})

        train_df = pd.read_parquet(model_conf.train_root_dir, columns=required_features)
        val_df = pd.read_parquet(model_conf.val_root_dir, columns=required_features)

        model.fit(train_df[model_conf.features], train_df['target'],
                  val_df[model_conf.features], val_df['target'], val_df['WEEK_NUM'])

        test_df = pd.read_parquet(options.test_data_root_dir, columns=required_features).reset_index(drop=True)

        preds, loss, auc = model.predict(test_df[model_conf.features], test_df['target'])
        print(f'Model {model_conf.model_name} - Test AUC: {auc} / Log Loss: {loss}')

        pred_series = pd.Series(preds)
        _, stability, _ = _stability_metric(test_df['WEEK_NUM'], test_df['target'], pred_series)

        print(f'Model {model_conf.model_name} - Test Stability Metric: {stability} ')
        eval_df = pd.DataFrame({'case_id': test_df['case_id'], 'WEEK_NUM': test_df['WEEK_NUM'], 'target': test_df['target'], 'pred': pred_series})

        if result_df is None:
            result_df = eval_df
            result_df['score'] = None
            result_df['score'] = result_df['pred']
        else:
            result_df = pd.merge(result_df, eval_df[['case_id', 'pred']], left_on='case_id', right_on='case_id', how='left')
            result_df['score'] = result_df['score'] + result_df['pred']

        result_df.drop(columns=['pred'], inplace=True)

        print('Saving model...')
        model.save(f'{options.best_model_output_path}_{model_conf.model_name}', f'{options.preprocess_json_path}_{model_conf.model_name}')

    result_df['score'] = result_df['score'] / len(models)

    auc_by_week_num = result_df.groupby('WEEK_NUM').apply(lambda x: roc_auc_score(x['target'].values, x['score'].values))
    print('Ensemble Model - Test AUC by WEEK_NUM')
    print(auc_by_week_num)

    plt.plot(auc_by_week_num.index, auc_by_week_num.values, 'b')
    plt.savefig(options.output_auc_png_path)

    loss = log_loss(result_df['target'].values, result_df['score'].values)
    auc = roc_auc_score(result_df['target'].values, result_df['score'].values)
    print(f'Ensemble Model - Test AUC: {auc} / Log Loss: {loss}')
    _, stability, _ = _stability_metric(result_df['WEEK_NUM'], result_df['target'], result_df['score'])
    print(f'Ensemble Model - Test Stability Metric: {stability} ')

    print('Saving results to the submission CSV file...')
    result_df[['case_id', 'score']].to_csv(options.submission_csv_file_path, index=False)


