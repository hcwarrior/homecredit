import json
import pickle
from typing import Dict

import optuna
from lightgbm import LGBMClassifier
import lightgbm
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV

from model.base_model import BaseModel

pd.set_option('future.no_silent_downcasting', True)


class LightGBM(BaseModel):
    def __init__(self,
                 transformations_by_feature: Dict[str, object] = None):
        super().__init__(transformations_by_feature)

    def _objective(self, trial,
                   df: pd.DataFrame, label_array: np.array,
                   val_df: pd.DataFrame, val_label_array: np.array):
        param = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "verbose": 50,
            "n_estimators": trial.suggest_int("n_estimators", 1000, 2000),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 24),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        }
        gbm = LGBMClassifier(**param)
        gbm.fit(df, label_array, eval_set=[(val_df, val_label_array)], eval_metric='auc',
                       callbacks=[lightgbm.log_evaluation(period=50),
                                  lightgbm.early_stopping(stopping_rounds=70)])
        preds = gbm.predict_proba(val_df)[:, 1]
        auroc = roc_auc_score(val_label_array, preds)
        return auroc

    def fit(self, df: pd.DataFrame, label_array: np.array,
        val_df: pd.DataFrame, val_label_array: np.array):
        print('Preprocessing...')
        df = self._preprocess(df)
        val_df = self._preprocess(val_df)

        # hyperparameter tuning
        print('Optimizing Hyperparameters...')
        optuna.logging.disable_default_handler()
        sampler = optuna.integration.SkoptSampler(
            skopt_kwargs={'n_random_starts': 5,
                          'acq_func': 'EI',
                          'acq_func_kwargs': {'xi': 0.02}})

        study = optuna.create_study(direction="maximize", sampler=sampler)
        _objective_currying = lambda trial: self._objective(trial, df, label_array, val_df, val_label_array)
        print('Optimizing...')
        study.optimize(_objective_currying, n_trials=60)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        best_params = trial.params
        print(best_params)
        ###

        print('Fitting...')
        monotone_constraints = None
        self.model = LGBMClassifier(boosting_type='gbdt', random_state=42,
                                    learning_rate=best_params['learning_rate'],
                                    max_depth=best_params['max_depth'],
                                    n_estimators=best_params['n_estimators'],
                                    colsample_bytree=best_params['colsample_bytree'],
                                    min_child_samples=best_params['min_child_samples'],
                                    num_leaves=best_params['num_leaves'],
                                    reg_alpha=best_params['reg_alpha'],
                                    reg_lambda=best_params['reg_lambda'],
                                    subsample=best_params['subsample'],
                                    monotone_constraints=monotone_constraints)
        self.model.fit(df, label_array, eval_set=[(val_df, val_label_array)], eval_metric='auc',
                       callbacks=[lightgbm.log_evaluation(period=5),
                                  # lightgbm.early_stopping(stopping_rounds=80)
                                  ])

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

