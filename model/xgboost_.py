import json
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import log_loss, auc, roc_auc_score

pd.set_option('future.no_silent_downcasting', True)


class XGBoost:
    def __init__(self,
                 transformations_by_feature: Dict[str, object] = None):
        self.transformations_by_feature = transformations_by_feature
        self.model = None

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, transformation in self.transformations_by_feature.items():
            type = transformation['type']
            prop = transformation['properties']

            if type == 'onehot':
                onehot = pd.DataFrame(np.zeros((len(df[col]), len(prop['vocab']))))
                for i, vocab in enumerate(prop['vocab']):
                    rows = df[col].index[df[col] == vocab]
                    onehot.loc[rows, i] = 1

                df = df.drop(columns=[col])
                df = pd.concat([df, onehot], axis=1)

            elif type == 'target_encoding':
                encoding_dict = dict(zip(prop['value'], prop['encoded']))
                df[col] = df[col].map(encoding_dict.get)

            elif type == 'binning':
                boundaries = [[float('-inf')] + prop['boundaries'] + [float('inf')]]
                for i in range(len(boundaries) - 1):
                    df[col][(df[col] >= boundaries[i]) & (df[col] < boundaries[i + 1])] = i
            elif type == 'standardization':
                df[col] = (df[col] - prop['mean']) / prop['stddev']
            else:
                pass

        return df

    def fit(self, df: pd.DataFrame, label_array: np.array,
        val_df: pd.DataFrame, val_label_array: np.array):
        print('Preprocessing...')
        df = self._preprocess(df)
        val_df = self._preprocess(val_df)

        print('Fitting...')
        train_mat = xgb.DMatrix(df.values, label_array)
        val_mat = xgb.DMatrix(val_df.values, val_label_array)
        evals = [(train_mat, 'train'), (val_mat, 'eval')]

        # negative : positive = 30 : 1
        base_param = {
            'learning_rate': 0.1,
            'tree_method': 'exact',
            'refresh_leaf': True,
            'max_depth': 5,
            'gamma': 0.6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'scale_pos_weight': 30,
            'reg_lambda': 3
        }
        update_param = base_param | {'updater': 'refresh', 'process_type': 'update'}
        params = base_param if self.model is None else update_param

        boosting_rounds = 400 if self.model is None else self.model.num_boosted_rounds()
        self.model = xgb.train(params, dtrain=train_mat, evals=evals, num_boost_round=boosting_rounds, early_stopping_rounds=100, xgb_model=self.model)

    def _preprocess_predict(self, df: pd.DataFrame):
        for col, transformation in self.transformations_by_feature.items():
            type = transformation['type']
            prop = transformation['properties']

            if type == 'onehot':
                onehot = pd.DataFrame(np.zeros((len(df[col]), len(prop['vocab']))))
                for i, vocab in enumerate(prop['vocab']):
                    rows = df[col].index[df[col] == vocab]
                    onehot.loc[rows, i] = 1

                df = df.drop(columns=[col])
                df = pd.concat([df, onehot], axis=1)
            elif type == 'target_encoding':
                encoding_dict = dict(zip(prop['value'], prop['encoded']))
                encoded = df[col].map(encoding_dict.get).fillna(0.0)
                df = df.drop(columns=[col])
                df.loc[:, col] = encoded
            elif type == 'binning':
                boundaries = [[float('-inf')] + prop['boundaries'] + [float('inf')]]
                for i in range(len(boundaries) - 1):
                    df[col][(df[col] >= boundaries[i]) & (df[col] < boundaries[i + 1])] = i
            elif type == 'standardization':
                standardized = (df[col] - prop['mean']) / prop['stddev']
                df = df.drop(columns=[col])
                df.loc[:, col] = standardized
            else:
                pass
        return df

    def predict(self, df_without_label: pd.DataFrame, label_array: np.ndarray = None):
        batch_size = 4096
        chunked_dfs = [df_without_label[i:i + batch_size].reset_index(drop=True) for i in range(0, len(df_without_label), batch_size)]

        preds = []
        for i in range(len(chunked_dfs)):
            chunked_df = self._preprocess_predict(chunked_dfs[i])
            test_mat = xgb.DMatrix(chunked_df.values)
            preds.append(self.model.predict(test_mat))

        loss, auroc = None, None
        pred = np.array([y for x in preds for y in x])
        if label_array is not None:
            loss = log_loss(label_array, pred)
            auroc = roc_auc_score(label_array, pred)

        return pred, loss, auroc

    def save(self, output_model_path: str, output_transformation_path: str):
        self.model.save_model(output_model_path)
        with open(output_transformation_path, 'w') as fd:
            json.dump(self.transformations_by_feature, fd)


    def load(self, input_model_path: str, input_transformation_path: str):
        self.model = xgb.Booster()
        self.model.load_model(input_model_path)

        with open(input_transformation_path, 'r') as fd:
            self.transformations_by_feature = json.load(fd)

