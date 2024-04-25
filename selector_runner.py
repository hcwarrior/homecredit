import gc
import json
import os
import time
import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset.feature.feature import *
from dataset.feature.util import optimize_dataframe 

from dataset.datainfo import RawInfo, RawReader, DATA_PATH
from dataset.feature.feature import *
from dataset.feature.util import optimize_dataframe
from dataset.const import TOPICS
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


def load_features(topic):
    with open(DATA_PATH / f'feature_definition/{topic}.json', 'r') as f:
        return [Feature.from_dict(feature) for feature in json.load(f).values()]


def load_data(topic, type_='train'):
    rawinfo = RawInfo()
    data = rawinfo.read_raw(topic, depth=1, reader=RawReader('polars'), type_=type_, stage='prep')
    base = rawinfo.read_raw('base', reader=RawReader('polars'), type_=type_)
    base = base.with_columns(pl.col('case_id').cast(pl.Int32))
    return data.join(base.select(['case_id', 'date_decision', 'target']), on='case_id', how='inner')

def load_feature_data(frame, features, verbose=False):
    query = [
        f'cast({feat.query} as {feat.agg.data_type}) as {feat.name}'
        for feat in features
    ]
    if verbose:
        for q in query:
            print(f'[*] Query: {q}')
    temp = pl.SQLContext(frame=frame).execute(
        f"""
        SELECT frame.case_id, frame.target
            , {', '.join(query)}
        from frame
        group by frame.case_id, frame.target""".replace(
            'float32', 'float'
        ),
        eager=True,
    )
    temp = optimize_dataframe(temp)
    return temp


def execute_query(frame, features, batch_size, type_='train'):
    start_time = time.time()
    for i, index in enumerate(tqdm(range(0, len(features), batch_size))):
        temp = load_feature_data(frame, features[index : index + batch_size])
        temp.write_parquet(
            DATA_PATH / f'{type_}_feature/{type_}_{topic}_features_{i}.parquet',
        )
        del temp
        gc.collect()
    print(f'[*] Elapsed time: {time.time() - start_time:.4f} sec')


def train_model(X: pd.DataFrame, y: pd.Series):
    cat_indicis = [i for i, c in enumerate(X.columns) if X[c].dtype == 'O']
    X = X.astype({c: 'category' for c in X.columns if X[c].dtype == 'O'})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.to_numpy().ravel(), test_size=0.2, random_state=42
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
        }
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_indicis,
    )
    train_auroc = (roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    test_auroc = (roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    print(f'Train AUC: {train_auroc:.4f}, Test AUC: {test_auroc:.4f}')
    return model

def select_features(df: pl.DataFrame) -> List[str]:
    y = df.select('target')
    X = df.drop(['case_id', 'target', 'case_id_right', 'case_id_right2']).to_pandas()
    model = train_model(X, y)
    features = X.columns[model.feature_importances_>1].to_list()
    return features

def read_select_df(path: str, features: List[str]) -> pl.DataFrame:
    df = pl.read_parquet(path)
    return df.select([c for c in df.columns if c in features] + ['case_id', 'target'])

if __name__ == '__main__':
    os.makedirs(DATA_PATH / 'feature_selection', exist_ok=True)
    batch_size = 1000

    for topic in TOPICS:
        if topic.depth == 1:
            print(f'[*] Selecting features for {topic.name}')
            selected_feature_list = []
            start_time = time.time()
            features = load_features(topic.name)
            frame = load_data(topic.name)
            for i, index in enumerate(tqdm(range(0, len(features), batch_size))):
                temp_path = DATA_PATH / f'parquet_files/train_feature/train_{topic.name}_features_{i}.parquet'
                if os.path.exists(temp_path):
                    temp = pl.read_parquet(temp_path)
                else:
                    temp = load_feature_data(frame, features[index : index + batch_size])
                    temp.write_parquet(temp_path)
                selected_features = select_features(temp)
                selected_feature_list += selected_features
                print(f'using {len(selected_features)}')
                gc.collect()
            with open(DATA_PATH / f'feature_selection/{topic.name}.json', 'w') as f:
                json.dump(selected_feature_list, f)
            print(f'[*] Elapsed time: {time.time() - start_time:.4f} sec')

if __name__ == '__main__':
    batch_size = 10
    for topic in TOPICS:
        if topic.depth == 1:
            with open(DATA_PATH / f'feature_selection/{topic.name}.json', 'r') as f:
                selected_feature_list = json.load(f)
            if len(selected_feature_list) < 1000:
                continue
            print(f'[*] Selecting features for {topic.name} secondary')
            secondary_selected_feature_list = []
            start_time = time.time()

            temp_path_ptn = f'parquet_files/train_feature/train_{topic.name}_features_*.parquet'
            file_list = sorted(list(DATA_PATH.glob(temp_path_ptn)))
            for i in tqdm(range(0, len(file_list), batch_size)):
                temp_file_list = file_list[i : i + batch_size]
                temp_base = read_select_df(temp_file_list[0], selected_feature_list)
                for file in temp_file_list[1:]:
                    temp = read_select_df(file, selected_feature_list)
                    temp_base = temp_base.join(temp, on=['case_id', 'target'], how='outer')
                    temp_base = temp_base.drop(['case_id_right', 'target_right'])
                    del temp
                    gc.collect()
                selected_features = select_features(temp_base)
                secondary_selected_feature_list += selected_features
                print(f'using {len(selected_features)} in {len(temp_base.columns)-2}')
                gc.collect()

            with open(DATA_PATH / f'feature_selection/{topic.name}_secondary.json', 'w') as f:
                json.dump(secondary_selected_feature_list, f)
            print(f'[*] Elapsed time: {time.time() - start_time:.4f} sec')
