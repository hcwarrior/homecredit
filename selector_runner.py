import gc
import json
import os
import polars as pl
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset.feature.feature import *
from dataset.feature.feature_loader import FeatureLoader

from dataset.datainfo import DATA_PATH
from dataset.feature.feature import *
from dataset.const import TOPICS
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


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
    X = df.drop(['case_id', 'target', 'case_id_right', 'case_id_right2']).to_pandas()
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


# if __name__ == '__main__':
#     SELECT_PATH = DATA_PATH / 'feature_selection'
#     os.makedirs(SELECT_PATH, exist_ok=True)
#     batch_size = 1000
#     postfix = ''

#     depth1_topics = [topic for topic in TOPICS if topic.depth == 1]
#     for topic in depth1_topics:
#         if os.path.exists(SELECT_PATH / f'{topic.name}{postfix}.json'):
#             continue

#         print(f'[*] Selecting features for {topic.name}')
#         selected_feature_list = []

#         # load already taken features
#         already_taken = sorted(list(SELECT_PATH.glob(f'{topic.name}_*{postfix}.json')))
#         for taken_file in already_taken:
#             selected_feature_list += read_json(taken_file)

#         fl = FeatureLoader(topic, type='train')
#         features = fl.load_features()
#         iter = fl.load_feature_data_batch(features, batch_size, skip=len(already_taken))
#         for i, temp_data in enumerate(iter):
#             if temp_data is None:
#                 continue

#             selected_temp = select_features(temp_data)
#             selected_feature_list += selected_temp
#             print(f'using {len(selected_temp)}')

#             del temp_data
#             gc.collect()
#             write_json(SELECT_PATH / f'{topic.name}_{i}{postfix}.json', selected_temp)
#         write_json(SELECT_PATH / f'{topic.name}{postfix}.json',selected_feature_list)

#         # delete teemp files
#         for i in range(len(features) // batch_size + 1):
#             temp_path = SELECT_PATH / f'{topic.name}_{i}{postfix}.json'
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)


if __name__ == '__main__':
    SELECT_PATH = DATA_PATH / 'feature_selection'
    os.makedirs(SELECT_PATH, exist_ok=True)
    batch_size = 1000
    postfix_preselected = '_secondary'
    postfix = '_tertiary'

    depth1_topics = [topic for topic in TOPICS if topic.depth == 1]
    for topic in depth1_topics:
        preselected = read_json(SELECT_PATH / f'{topic.name}{postfix_preselected}.json')
        if len(preselected) < 1000:
            continue
        if os.path.exists(SELECT_PATH / f'{topic.name}{postfix}.json'):
            continue

        print(f'[*] Selecting features for {topic.name}')
        selected_feature_list = []

        # load already taken features
        already_taken = sorted(list(SELECT_PATH.glob(f'{topic.name}_*{postfix}.json')))
        for taken_file in already_taken:
            selected_feature_list += read_json(taken_file)

        fl = FeatureLoader(topic, type='train')
        features = fl.load_features(preselected)
        iter = fl.load_feature_data_batch(features, batch_size, skip=len(already_taken))
        for i, temp_data in enumerate(iter):
            if temp_data is None:
                continue

            selected_temp = select_features(temp_data)
            selected_feature_list += selected_temp
            print(f'using {len(selected_temp)}')

            del temp_data
            gc.collect()
            write_json(SELECT_PATH / f'{topic.name}_{i}{postfix}.json', selected_temp)
        write_json(SELECT_PATH / f'{topic.name}{postfix}.json', selected_feature_list)

        # delete teemp files
        for i in range(len(features) // batch_size + 1):
            temp_path = SELECT_PATH / f'{topic.name}_{i}{postfix}.json'
            if os.path.exists(temp_path):
                os.remove(temp_path)
