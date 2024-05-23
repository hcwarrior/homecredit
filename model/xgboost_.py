import json
import pickle
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV

from model.base_model import BaseModel

pd.set_option('future.no_silent_downcasting', True)


class XGBoost(BaseModel):
    def __init__(self,
                 transformations_by_feature: Dict[str, object] = None):
        super().__init__(transformations_by_feature)


    def fit(self, df: pd.DataFrame, label_array: np.array,
        val_df: pd.DataFrame, val_label_array: np.array):
        print('Preprocessing...')
        df = self._preprocess(df)
        val_df = self._preprocess(val_df)

        # print('Grid Searching')
        # params = {'max_depth': [5, 7], 'min_child_weight': [1, 3], 'colsample_bytree': [0.5, 0.75]}
        # grid_model = xgb.XGBClassifier(tree_method='hist', enable_categorical=True, n_estimators=100,
        #                                learning_rate=0.05, reg_alpha=0.05, scale_pos_weight=30)
        # gridcv = GridSearchCV(grid_model, param_grid=params, cv=3)
        # gridcv.fit(df, label_array, eval_set=[(val_df, val_label_array)],
        #            early_stopping_rounds=30, eval_metric='auc')
        # best_params = gridcv.best_params_

        best_params = {'max_depth': 5, 'min_child_weight': 3, 'colsample_bytree': 0.75}

        print('Fitting...')
        monotone_constraints = None
        self.model = xgb.XGBClassifier(tree_method='hist', enable_categorical=True, max_depth=best_params['max_depth'], n_estimators=1000,
                                       min_child_weight=best_params['min_child_weight'],
                                       colsample_bytree=best_params['colsample_bytree'], colsample_bylevel=0.8, random_state=42,
                                       learning_rate=0.05, reg_alpha=0.05, scale_pos_weight=30,
                                       monotone_constraints=monotone_constraints)
        self.model.fit(df, label_array, eval_set=[(val_df, val_label_array)],
                       early_stopping_rounds=50, eval_metric='auc', verbose=5)

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

