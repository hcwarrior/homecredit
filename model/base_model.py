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

        return df

    def fit(self, df: pd.DataFrame, label_array: np.array,
        val_df: pd.DataFrame, val_label_array: np.array):
        raise NotImplementedError("Please Implement this method")


    def predict(self, df_without_label: pd.DataFrame, label_array: np.ndarray = None):
        raise NotImplementedError("Please Implement this method")


    def save(self, output_model_path: str, output_transformation_path: str):
        raise NotImplementedError("Please Implement this method")


    def load(self, input_model_path: str, input_transformation_path: str):
        raise NotImplementedError("Please Implement this method")

