from contextlib import contextmanager
import gc
import json
import os
import time
import polars as pl
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
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
    X = df.drop(['case_id', 'target', 'MONTH', 'case_id_right', 'case_id_right2']).to_pandas()
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

def objective(trial, X_train, X_test, y_train, y_test, cat_indicis):
    model = LGBMClassifier(
        **{
            'n_estimators': trial.suggest_int('n_estimators', 100, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-2, 1e-1),
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            'verbose': -1,
            'random_state': 42,
            'is_unbalance': True,
            'importance_type': 'gain',
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
            'min_child_samples': trial.suggest_int("min_child_samples", 10, 100),
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
    gc.collect()
    return test_auroc

def X_y_split(base: pl.DataFrame):
    if 'case_id_right' in base.columns:
        raise ValueError('case_id_right already exists in base')
    y = base.select('target').to_pandas()
    X = base.drop(['case_id', 'target', 'MONTH']).to_pandas()
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

    ############################################################################################
    # selection again
    ############################################################################################
    # raw_info = RawInfo()
    # base = raw_info.read_raw('base', reader=RawReader('polars'), type_='train')
    # base = base.select([pl.col('case_id').cast(pl.Int32), 'target', 'MONTH'])

    # select_topic = ['applprev', 'person']
    # depth1_topics = [
    #     topic for topic in TOPICS if topic.depth == 1 and topic.name in select_topic
    # ]
    # for topic in depth1_topics:
    #     print(f'[*] Processing {topic.name}...')
    #     selected = read_json(SELECT_PATH / f'selected_features.json')
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
    # selection_again_features_1 = select_features(base)

    # raw_info = RawInfo()
    # base = raw_info.read_raw('base', reader=RawReader('polars'), type_='train')
    # base = base.select([pl.col('case_id').cast(pl.Int32), 'target', 'MONTH'])

    # select_topic = ['credit_bureau_a','credit_bureau_b']
    # depth1_topics = [
    #     topic for topic in TOPICS if topic.depth == 1 and topic.name in select_topic
    # ]
    # for topic in depth1_topics:
    #     print(f'[*] Processing {topic.name}...')
    #     selected = read_json(SELECT_PATH / f'selected_features.json')
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
    # selection_again_features_2 = select_features(base)

    # ############################################################################################
    # # data preparation
    # ############################################################################################
    # raw_info = RawInfo()
    # base = raw_info.read_raw('base', reader=RawReader('polars'), type_='train')
    # base = base.select([pl.col('case_id').cast(pl.Int32), 'target', 'MONTH'])
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
    #     selected = read_json(SELECT_PATH / f'selected_features.json')
    #     if topic.name in ('applprev', 'person'):
    #         selected = [c for c in selected if c in selection_again_features_1]
    #     elif topic.name in ('credit_bureau_a', 'credit_bureau_b'):
    #         selected = [c for c in selected if c in selection_again_features_2]
    #     else:
    #         pass
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

    # base.write_parquet(DATA_PATH / 'train_all.parquet')

    ############################################################################################
    # hyperopt
    ############################################################################################
    base = pl.read_parquet(DATA_PATH / 'train_all.parquet')
    X, y, cat_indicis = X_y_split(base)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.to_numpy().ravel(), test_size=0.2, random_state=42
    )
    del X, y, base
    gc.collect()

    study_name = 'lgbm'
    MODEL_PATH = DATA_PATH / 'model' / study_name
    os.makedirs(DATA_PATH / 'model' / study_name, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{MODEL_PATH}/study.db",
        load_if_exists=True,
    )
    trials_taken = len(study.get_trials())

    n_trials = 120
    if trials_taken >= n_trials:
        print(f"Already taken {trials_taken} trials. Skip hyperopt")
    else:
        n_trials = n_trials - trials_taken
        study.optimize(
            lambda trial: objective(trial, X_train, X_test, y_train, y_test, cat_indicis),
            n_trials=n_trials,
            show_progress_bar=True,
        )

    ############################################################################################
    # train model with best params
    ############################################################################################
    best_params = study.best_trial.params
    basic_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        'verbose': -1,
        'random_state': 42,
        'is_unbalance': True,
    }
    basic_params.update(best_params)
    print(basic_params)

    model = LGBMClassifier(**basic_params)
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_indicis,
    )
    model.booster_.save_model(MODEL_PATH / 'model.pkl')
    artifacts = {'features': X_train.columns.to_list(), 'cat_indicis': cat_indicis}
    write_json(MODEL_PATH / 'artifacts.json', artifacts)

    # load model
    model = lgb.LGBMClassifier()
    model = lgb.Booster(model_file=MODEL_PATH / 'model.pkl')
    artifacts = read_json(MODEL_PATH / 'artifacts.json')
    base = pl.read_parquet(DATA_PATH / 'train_all.parquet')
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
    
