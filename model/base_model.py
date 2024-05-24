import json
import pickle
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import log_loss, auc, roc_auc_score

pd.set_option('future.no_silent_downcasting', True)


class BaseModel:
    def __init__(self,
                 transformations_by_feature: Dict[str, object] = None):
        self.transformations_by_feature = transformations_by_feature
        self.model = None

    def stability_metric(self, week, y_true, y_pred):
        """
        Custom metric for model optimization during training
        """
        weeks_to_score = week
        gini_in_time = []
        print((week, y_true, y_pred))

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

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, transformation in self.transformations_by_feature.items():
            type = transformation['type']
            prop = transformation.get('properties', set())

            if type == 'onehot':
                onehot = pd.DataFrame(np.zeros((len(df[col]), len(prop['vocab']))))
                for i, vocab in enumerate(prop['vocab']):
                    rows = df[col].index[df[col] == vocab]
                    onehot.loc[rows, i] = 1

                df = df.drop(columns=[col])
                df = pd.concat([df, onehot], axis=1)
            elif type == 'target_encoding':
                encoding_dict = dict(zip(prop['value'], prop['encoded']))
                encoded = df[col].map(encoding_dict.get).astype('float64').fillna(0.0)
                df[col] = encoded
            elif type == 'binning':
                boundaries = [[float('-inf')] + prop['boundaries'] + [float('inf')]]
                for i in range(len(boundaries) - 1):
                    df[col][(df[col] >= boundaries[i]) & (df[col] < boundaries[i + 1])] = i
            elif type == 'standardization':
                df[col] = (df[col] - prop['mean']) / prop['stddev']
            elif type == 'categorical':
                df[col] = df[col].astype('category')
            else:
                pass

        # to ensure WEEK_NUM is the first
        if 'WEEK_NUM' in df.columns:
            return df[['WEEK_NUM'] + [col for col in df.columns if col != 'WEEK_NUM']]

        return df

    def fit(self, df: pd.DataFrame, label_array: np.array,
        val_df: pd.DataFrame, val_label_array: np.array, val_week_num: np.array):
        raise NotImplementedError("Please Implement this method")


    def predict(self, df_without_label: pd.DataFrame, label_array: np.ndarray = None):
        raise NotImplementedError("Please Implement this method")


    def save(self, output_model_path: str, output_transformation_path: str):
        raise NotImplementedError("Please Implement this method")


    def load(self, input_model_path: str, input_transformation_path: str):
        raise NotImplementedError("Please Implement this method")

