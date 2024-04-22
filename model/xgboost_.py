import math
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import log_loss, auc
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


class XGBoost:
    def __init__(self,
                 transformations_by_feature: Dict[str, object]):
        self.transformations_by_feature = transformations_by_feature
        self.preprocessing_by_col = {}
        self.model = None


    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, transformation in self.transformations_by_feature.items():
            type = transformation['type']
            prop = transformation['properties']

            if type == 'onehot':
                ohe = OneHotEncoder(sparse_output=False)
                transformed = ohe.fit_transform(df[col].values.reshape(-1, 1))
                onehot_df = pd.DataFrame(transformed, columns=ohe.get_feature_names_out([col]))

                df = df.drop(columns=[col])
                df = pd.concat([df, onehot_df], axis=1)

                self.preprocessing_by_col[col] = ohe
            elif type == 'target_encoding':
                df[col] = df[col].replace(dict(zip(prop['value'], prop['encoded'])))
                self.preprocessing_by_col[col] = prop
            elif type == 'binning':
                boundaries = [[float('-inf')] + prop['boundaries'] + [float('inf')]]
                for i in range(len(boundaries) - 1):
                    df[col][(df[col] >= boundaries[i]) & (df[col] < boundaries[i + 1])] = i
                self.preprocessing_by_col[col] = prop
            elif type == 'standardization':
                df[col] = (df[col] - prop['mean']) / prop['stddev']
                self.preprocessing_by_col[col] = prop
            else:
                # pass
                self.preprocessing_by_col[col] = {}

        return df


    def fit(self, df: pd.DataFrame, label_array: np.array):
        df = self._preprocess(df)
        train_mat = xgb.DMatrix(df.values, label_array)

        classifier = XGBClassifier(objective='binary:logistic')
        self.model = classifier.fit({
            'learning_rate': 0.005,
            'update': 'refresh',
            'process_type': 'update',
            'refresh_leaf': True,
            'reg_lambda': 3,  # L2
            #'reg_alpha': 3,  # L1
            'silent': False,
        }, dtrain=train_mat, xgb_model=self.model)


    def _preprocess_predict(self, df: pd.DataFrame):
        for col, transformation in self.transformations_by_feature.items():
            type = transformation['type']
            prop = self.preprocessing_by_col[col]

            if type == 'onehot':
                ohe = prop
                onehot_df = ohe.transform(df[col].values.reshape(-1, 1))
                df = df.drop(columns=[col])
                df = pd.concat([df, onehot_df], axis=1)
            elif type == 'target_encoding':
                df[col] = df[col].replace(dict(zip(prop['value'], prop['encoded'])))
            elif type == 'binning':
                boundaries = [[float('-inf')] + prop['boundaries'] + float('inf')]
                for i in range(len(boundaries) - 1):
                    df[col][(df[col] >= boundaries[i]) & (df[col] < boundaries[i + 1])] = i
            elif type == 'standardization':
                df[col] = (df[col] - prop['mean']) / prop['stddev']
            else:
                pass
        return df

    def predict(self, df: pd.DataFrame, label: str):
        df = self._preprocess_predict(df)
        test_mat = xgb.DMatrix(df.drop(columns=[label]).values)
        pred = self.model.predict_proba(test_mat)

        loss = log_loss(df[label].values, pred)
        auroc = auc(df[label].values, pred)

        return loss, auroc



