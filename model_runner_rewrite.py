from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import time
import polars as pl
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from dataset.feature.feature import *
from dataset.feature.feature_loader import FeatureLoader
from dataset.feature.util import optimize_dataframe

from dataset.datainfo import RawInfo, RawReader, DATA_PATH
from dataset.const import TOPICS
import lightgbm as lgb


def train_model(X: pd.DataFrame, y: pd.Series):
    cat_indicis = [i for i, c in enumerate(X.columns) if X[c].dtype == 'O']
    X = X.astype({c: 'category' for c in X.columns if X[c].dtype == 'O'})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.to_numpy().ravel(), test_size=0.2, random_state=43
    )
    model = LGBMClassifier(
        **{
            'n_estimators': 200,
            'max_depth': 3,
            'subsample': 0.7,
            'learning_rate': 0.01,
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True,
            'importance_type': 'gain',
        }
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_indicis,
    )
    train_auroc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f'Train AUC: {train_auroc:.4f}, Test AUC: {test_auroc:.4f}')
    del X_train, X_test, y_train, y_test
    return model


def select_features(df: pl.DataFrame) -> List[str]:
    y = df.select('target')
    X = df.drop(
        ['case_id', 'target', 'date_decision', 'case_id_right', 'case_id_right2']
    ).to_pandas()
    model = train_model(X, y)
    features = X.columns[model.feature_importances_ > 0].to_list()
    del X, y
    return features


def read_json(path: str) -> List[str]:
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path: str, data: List[str]):
    with open(path, 'w') as f:
        json.dump(data, f)


class StabilityEvalMetric:

    def __init__(self, weeks_to_score: pd.Series) -> None:
        self.weeks_to_score = weeks_to_score

    def __call__(self, y_true, y_pred, verbose=False): 
        gini_in_time = []

        for week in sorted(self.weeks_to_score.unique()):
            week_idx = self.weeks_to_score.eq(week)
            gini = np.array(2 * roc_auc_score(y_true[week_idx], y_pred[week_idx]) - 1)
            gini_in_time.append(gini)

        W_FALLINGRATE = 88.0
        W_RESSTD = -0.5
        x = np.arange(len(gini_in_time))
        y = np.array(gini_in_time)
        a, b = np.polyfit(x, y, 1)
        y_hat = a * x + b
        residuals = y - y_hat
        res_std = np.std(residuals)
        avg_gini = np.mean(y)
        stability_score = avg_gini + W_FALLINGRATE * min(0, a) + W_RESSTD * res_std
        if verbose:
            print('logic : avg_gini + 88.0 * min(0, a) + -0.5 * res_std')
            print(f'avg_auroc: {avg_gini/2+0.5:.4f}')
            print(f'avg_gini: {avg_gini:.4f}, falling_rate: {a:.4f}, res_std: {res_std:.4f}')
            print(f'stability_score: {stability_score:.4f}')
        is_higher_better = True

        return 'stability_score', stability_score, is_higher_better


def lgbm_objective(trial, X_train, X_test, y_train, y_test, cat_indicis, y_ratio, week_to_score):
    """
    LGBMClassifier parameters search
    """

    params = {
        'n_estimators': 5000,
        'num_leaves': trial.suggest_int('num_leaves', 2, 10),
        'min_child_samples': trial.suggest_int('min_data_in_leaf', 2000, 8000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08),
        'reg_alpha': trial.suggest_float('lambda_l1', 1e-3, 1, log=True),
        'reg_lambda': trial.suggest_float('lambda_l2', 1e-3, 1, log=True),
        'colsample_bytree': trial.suggest_float('feature_fraction', 0.3, 1),
        'subsample': trial.suggest_float('bagging_fraction', 0.8, 1),
        'subsample_freq': trial.suggest_int('bagging_freq', 0, 10),
        'objective': 'binary',
        'metric': 'None',
        'verbosity': -1,
        'random_state': 42,
        'scale_pos_weight': y_ratio,
        # 'device': 'gpu',
        'max_bin': 255,
        'n_jobs': -1,
    }
    model = LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_indicis,
        eval_set=[(X_test, y_test)],
        eval_metric=StabilityEvalMetric(week_to_score),
        callbacks=[early_stopping(100), log_evaluation(100)],
    )
    y_pred = model.predict_proba(X_test)[:, 1]
    _, stability_score, _ = StabilityEvalMetric(week_to_score)(
        np.array(y_test), np.array(y_pred)
    )
    del X_test, X_train, y_test, y_train, model, y_pred, week_to_score, cat_indicis, y_ratio
    return stability_score


def X_y_split(base: pl.DataFrame):
    if 'case_id_right' in base.columns:
        raise ValueError('case_id_right already exists in base')
    y = base.select('target').to_pandas()
    X = base.drop(['case_id', 'target', 'date_decision', 'week_group', 'WEEK_NUM']).to_pandas()
    X = X.astype({c: 'category' for c in X.columns if X[c].dtype == 'O'})
    cat_indicis = [i for i, c in enumerate(X.columns) if X[c].dtype == 'category']
    return X, y, cat_indicis

@contextmanager
def timechecker(msg):
    start = time.time()
    yield
    print(f'[{msg}] elapsed time: {time.time() - start:.2f} sec')


if __name__ == '__main__':
    SELECT_PATH = DATA_PATH / 'feature_selection'

    # ############################################################################################
    # # data preparation
    # ############################################################################################
    # raw_info = RawInfo()
    # base = raw_info.read_raw('base', reader=RawReader('polars'), type_='train')
    # base = base.select([pl.col('case_id').cast(pl.Int32), 'target', 'date_decision', 'WEEK_NUM'])
    # depth0_topics = [topic for topic in TOPICS if topic.depth == 0]
    # for topic in depth0_topics:
    #     print(f'[*] Processing {topic.name}...')
    #     data = raw_info.read_raw(topic.name, reader=RawReader('polars'), type_='train')
    #     data = optimize_dataframe(data)
    #     base = base.join(data, on='case_id', how='left')
    #     del data

    # depth1_topics = [topic for topic in TOPICS if topic.depth == 1]
    # for topic in depth1_topics:
    #     print(f'[*] Processing {topic.name}...')
    #     selected = read_json(SELECT_PATH / f'selected_features_final.json')
    #     fl = FeatureLoader(topic, type='train')
    #     features = fl.load_features(selected)
    #     data = fl.load_feature_data(features)
    #     data = data.drop('target')
    #     dup_keyword = '_if_1_eq_1_then_num_group1_'
    #     dupable_col = [c for c in data.columns if dup_keyword in c]
    #     print(dupable_col)
    #     data = data.rename({col: f'{col}_{topic.name}' for col in dupable_col})
    #     print('shape:', data.shape)
    #     data = optimize_dataframe(data)
    #     base = base.join(data, on='case_id', how='left')
    #     del data

    # date_cols = [c for c in base.columns if str(base[c].dtype) == 'String' and (c.endswith('d__') or c.endswith('D'))]
    # for c in date_cols:
    #     base = base.with_columns(
    #         ((pl.col('date_decision').cast(pl.Date) - pl.col(c).cast(pl.Date)).fill_null(0).cast(pl.Int64) / 86400000).alias(c)
    #     )
    # base = optimize_dataframe(base, verbose=True)
    # base.write_parquet(DATA_PATH / 'train_all_20240428.parquet')

    # features = [c for c in base.columns if c not in ['case_id', 'target', 'date_decision']]
    # batch_size = 600
    # selection = []
    # for i in range(0, len(features), batch_size):
    #     data = base.select(['case_id', 'target', 'date_decision'] + features[i:i + batch_size])
    #     selection += select_features(data)
    # base = base.select(['case_id', 'target', 'date_decision'] + selection)
    # base.write_parquet(DATA_PATH / 'train_all_20240428_1.parquet')

    ############################################################################################
    # hyperopt
    ############################################################################################

    N_SPLIT = 5

    for i in range(N_SPLIT):
        if i <= 1:
            continue

        # set modeling data
        base = pl.read_parquet(DATA_PATH / 'train_all_20240428_1.parquet')
        raw_info = RawInfo()
        raw = raw_info.read_raw('base', reader=RawReader('polars'), type_='train')
        raw = raw.select(
            [
                pl.col('case_id').cast(pl.Int32),
                (pl.col('WEEK_NUM') / 92 * N_SPLIT)
                .floor()
                .alias('week_group')
                .cast(pl.Int32),
                pl.col('WEEK_NUM'),
            ]
        )
        base = base.join(raw, on='case_id', how='left')

        base_train = base.filter(pl.col('week_group') == i)
        base_test = base.filter(pl.col('week_group') != i)
        X_train, y_train, cat_indicis = X_y_split(base_train)
        X_test, y_test, _ = X_y_split(base_test)
        y_train, y_test = y_train.to_numpy().ravel(), y_test.to_numpy().ravel()
        y_ratio = np.sum(y_train == 0, axis=0) / np.sum(y_train == 1, axis=0)
        week_to_score = (
            base_test.select(['WEEK_NUM'])
            .to_pandas()['WEEK_NUM']
            .reset_index(drop=True)
        )

        del base, base_train, base_test, raw
        gc.collect()

        # hyper parameter optimization
        study_name = f'timesplit_{i}'
        MODEL_PATH = Path('output') / 'model' / study_name
        os.makedirs(MODEL_PATH, exist_ok=True)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=f"sqlite:///{MODEL_PATH}/study.db",
            load_if_exists=True,
        )
        trials_taken = len(study.get_trials())

        n_trials = 50
        if trials_taken >= n_trials:
            print(f"Already taken {trials_taken} trials. Skip hyperopt")
        else:
            n_trials = n_trials - trials_taken
            study.optimize(
                lambda trial: lgbm_objective(
                    trial,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    cat_indicis,
                    y_ratio,
                    week_to_score,
                ),
                n_trials=n_trials,
                show_progress_bar=True,
            )

        # train model with best params
        best_params = study.best_trial.params
        basic_params = {
            'n_estimators': 5000,
            'objective': 'binary',
            'metric': 'None',
            'verbosity': -1,
            'random_state': 42,
            'scale_pos_weight': y_ratio,
            'max_bin': 255,
            'n_jobs': -1,
        }
        basic_params.update(best_params)
        print(basic_params)
        write_json(MODEL_PATH / 'best_params.json', basic_params)

        model = LGBMClassifier(**basic_params)
        model.fit(
            X_train,
            y_train,
            categorical_feature=cat_indicis,
            eval_set=[(X_test, y_test)],
            eval_metric=StabilityEvalMetric(week_to_score),
            callbacks=[early_stopping(100), log_evaluation(100)],
        )
        model.booster_.save_model(MODEL_PATH / 'model.pkl')
        artifacts = {'features': X_train.columns.to_list(), 'cat_indicis': cat_indicis}
        write_json(MODEL_PATH / 'artifacts.json', artifacts)

        del X_train, X_test, y_train, y_test, cat_indicis, y_ratio, week_to_score

# 알맞은 앙상블
# 변수 선택(안정성)


    ############################################################################################
    # train model with best params
    ############################################################################################

    # load model
    model = lgb.LGBMClassifier()
    model = lgb.Booster(model_file=MODEL_PATH / 'model.pkl')
    artifacts = read_json(MODEL_PATH / 'artifacts.json')
    base = pl.read_parquet(DATA_PATH / 'train_all_20240428_1.parquet')
    X, y, _ = X_y_split(base)
    X = X[artifacts['features']]
    X = X.astype({c: 'category' for i, c in enumerate(X.columns) if i in artifacts['cat_indicis']})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.to_numpy().ravel(), test_size=0.2, random_state=42
    )
    # del X, y, base
    # gc.collect()

    train_auroc = roc_auc_score(y_train, model.predict(X_train))
    test_auroc = roc_auc_score(y_test, model.predict(X_test))
    print(f'Train AUC: {train_auroc:.4f}, Test AUC: {test_auroc:.4f}')

