import json
import pickle
from typing import Dict

from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
import lightgbm
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV

from model.base_model import BaseModel

pd.set_option('future.no_silent_downcasting', True)


class CatBoost(BaseModel):
    def __init__(self,
                 transformations_by_feature: Dict[str, object] = None):
        super().__init__(transformations_by_feature)

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
                df[col] = df[col].astype('str')
            else:
                pass

        return df

    def fit(self, df: pd.DataFrame, label_array: np.array,
        val_df: pd.DataFrame, val_label_array: np.array):
        print('Preprocessing...')
        df = self._preprocess(df)
        val_df = self._preprocess(val_df)

        print('Grid Searching')
        # params = {'max_depth': [5, 7], 'n_estimators': [100, 500, 1000], 'colsample_bytree': [0.5, 0.75]}

        # grid_model = LGBMClassifier(boosting_type='gbdt', random_state=42, early_stopping_rounds=50)
        # gridcv = GridSearchCV(grid_model, param_grid=params, verbose=2, n_jobs=-1, cv=3)
        # gridcv.fit(df, label_array, eval_set=[(val_df, val_label_array)],
        #            eval_metric='auc',
        #            callbacks=[lightgbm.log_evaluation(period=5),
        #                       lightgbm.early_stopping(stopping_rounds=30)])
        # best_params = gridcv.best_params_

        best_params = {'max_depth': 5, 'iterations': 1000, 'colsample_bylevel': 0.75}

        cat_cols = [col for col, transformation in self.transformations_by_feature.items() if transformation['type'] == 'categorical']

        train_pool = Pool(df, label_array, cat_features=cat_cols)
        val_pool = Pool(val_df, val_label_array, cat_features=cat_cols)

        print('Fitting...')
        self.model = CatBoostClassifier(
            eval_metric='AUC',
            learning_rate=0.07,
            iterations=best_params['iterations'],
            colsample_bylevel=best_params['colsample_bylevel'],
            max_depth=best_params['max_depth'])
        self.model.fit(train_pool, eval_set=val_pool, verbose=5, early_stopping_rounds=50)


    def predict(self, df_without_label: pd.DataFrame, label_array: np.ndarray = None):
        batch_size = 4096
        chunked_dfs = [df_without_label[i:i + batch_size].reset_index(drop=True) for i in range(0, len(df_without_label), batch_size)]

        preds = []
        for i in range(len(chunked_dfs)):
            chunked_df = self._preprocess(chunked_dfs[i])
            preds.append(self.model.predict_proba(chunked_df)[:, 1])

        loss, auroc = None, None
        pred = np.concatenate(preds)
        if label_array is not None:
            loss = log_loss(label_array, pred)
            auroc = roc_auc_score(label_array, pred)

        return pred, loss, auroc

    def save(self, output_model_path: str, output_transformation_path: str):
        with open(output_model_path, "wb") as fd:
            pickle.dump(self.model, fd)
        with open(output_transformation_path, 'w') as fd:
            json.dump(self.transformations_by_feature, fd)

    def load(self, input_model_path: str, input_transformation_path: str):
        with open(input_model_path, "rb") as fd:
            self.model = pickle.load(fd)

        with open(input_transformation_path, 'r') as fd:
            self.transformations_by_feature = json.load(fd)

