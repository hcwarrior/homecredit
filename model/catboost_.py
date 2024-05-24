import json
import pickle
from typing import Dict

import optuna
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
import lightgbm
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV

from model.base_model import BaseModel

pd.set_option('future.no_silent_downcasting', True)


class CatBoostEvalMetricStability:
    def __init__(self, week_num, cat_boost):
        self.week_num = week_num
        self.cat_boost = cat_boost

    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        # the larger metric value the better
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        preds = np.array(approxes[0])
        target = np.array(target)

        metric = self.cat_boost.stability_metric(self.week_num, target, preds)
        return metric, 0

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

        # to ensure WEEK_NUM is the first
        if 'WEEK_NUM' in df.columns:
            return df[['WEEK_NUM'] + [col for col in df.columns if col != 'WEEK_NUM']]

        return df

    def _objective(self, trial,
                   train_pool, val_pool,
                   val_df, val_label_array, val_week_num):
        param = {
            "eval_metric": "AUC",
            # "eval_metric": CatBoostEvalMetricStability(val_week_num, self),
            "task_type": "GPU",
            "devices": "0",
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "iterations": trial.suggest_int("iterations", 1000, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        print(param)
        gbm = CatBoostClassifier(**param)
        gbm.fit(train_pool, eval_set=val_pool, verbose=50, early_stopping_rounds=70)
        preds = gbm.predict_proba(val_df)[:, 1]
        metric = roc_auc_score(val_label_array, preds)
        # metric = self.stability_metric(val_week_num, val_label_array, preds)
        return metric

    def fit(self, df: pd.DataFrame, label_array: np.array,
        val_df: pd.DataFrame, val_label_array: np.array, val_week_num: np.array):
        print('Preprocessing...')
        df = self._preprocess(df)
        val_df = self._preprocess(val_df)

        monotone_constraints = None
        cat_cols = [col for col, transformation in self.transformations_by_feature.items() if transformation['type'] == 'categorical']

        train_pool = Pool(df, label_array, cat_features=cat_cols)
        val_pool = Pool(val_df, val_label_array, cat_features=cat_cols)

        # hyperparameter tuning
        # print('Optimizing Hyperparameters...')
        # optuna.logging.disable_default_handler()
        # sampler = optuna.integration.SkoptSampler(
        #     skopt_kwargs={'n_random_starts': 5,
        #                   'acq_func': 'EI',
        #                   'acq_func_kwargs': {'xi': 0.02}})
        #
        # study = optuna.create_study(direction="maximize", sampler=sampler)
        # _objective_currying = lambda trial: self._objective(trial, train_pool, val_pool, val_df, val_label_array, val_week_num)
        # print('Optimizing...')
        # study.optimize(_objective_currying, n_trials=5)
        #
        # print("Number of finished trials: {}".format(len(study.trials)))
        # print("Best trial:")
        # trial = study.best_trial
        #
        # print("  Value: {}".format(trial.value))
        # print("  Params: ")
        # best_params = trial.params
        # print(best_params)
        ###
        best_params = {'max_depth': 7, 'iterations': 1997, 'min_child_samples': 90, 'learning_rate': 0.048221020103455366, 'reg_lambda': 6.30114054634975}

        print('Fitting...')

        self.model = CatBoostClassifier(
            eval_metric='AUC',
            task_type='GPU',
            devices='0',
            max_depth=best_params['max_depth'],
            iterations=best_params['iterations'],
            min_child_samples=best_params['min_child_samples'],
            learning_rate=best_params['learning_rate'],
            reg_lambda=best_params['reg_lambda'],
            monotone_constraints=monotone_constraints)
        self.model.fit(train_pool, eval_set=val_pool, verbose=5)


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

