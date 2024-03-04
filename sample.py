import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import sklearn.metrics
from sklearn.model_selection import train_test_split
from dataset.datainfo import RawInfo
from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
import warnings
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def get_config():
    base_path = os.getcwd()
    data_path = os.path.join(base_path, "home-credit-credit-risk-model-stability")

    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default=base_path)
    parser.add_argument("--data_path", type=str, default=data_path)
    parser.add_argument("--raw_format", type=str, default="parquet")

    return parser.parse_args()


def prepare_base_data(conf: Namespace = None, type_: str = "train"):
    print("prepare_base_data ...")
    infos = RawInfo(conf)
    base_df = infos.read_raw("base", type_=type_)
    static_df = infos.read_raw("static", depth=0, type_=type_)
    static_cb_df = infos.read_raw("static_cb", depth=0, type_=type_)

    joined_df = pd.merge(base_df, static_df, on="case_id", how="left", suffixes=("_base", "_static"))
    joined_df = pd.merge(joined_df, static_cb_df, on="case_id", how="left", suffixes=("", "_static_cb"))
    print(f"base shape: {base_df.shape} & static shape: {static_df.shape} & static_cb shape: {static_cb_df.shape} & joined shape: {joined_df.shape}")

    return joined_df


def devval(df):
    conditions = [
        df["MONTH"].between(201909, 202008),
        df["MONTH"].between(201901, 201908)
    ]
    choices = [0, 1]
    df['devval'] = np.select(conditions, choices, default=2)


def get_tree_selector(
        df: pd.DataFrame,
        target: str,
        n_estimators: int = 5,
        max_features: int = None,
) -> SelectFromModel:
    print("select_features ...")
    X = df.drop(target, axis=1)
    y = df[target]

    clf = ExtraTreesClassifier(n_estimators=n_estimators)
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True, max_features=max_features)
    return model


def exclude_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("exclude_object_columns ...")
    return df.select_dtypes(exclude=["object"])


def objective(trial, X, y):
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }

    """
        Trial 40 finished with value: 0.7972759296627593 and parameters: 
        {'lambda_l1': 1.2241636481438304e-05,
        'lambda_l2': 9.985408160774956,
        'num_leaves': 211,
        'feature_fraction': 0.5443119666214574,
        'bagging_fraction': 0.8414338881950802,
        'bagging_freq': 5,
        'min_child_samples': 70}. Best is trial 40 with value: 0.7972759296627593."""

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    auroc = sklearn.metrics.roc_auc_score(valid_y, preds)
    return auroc


def inference(selector, model, X):
    X_sel = selector.transform(
        exclude_object_columns(X)
        .fillna(-99999999)
        .drop("target", axis=1))
    
    pred = model.predict(X_sel)
    auroc = sklearn.metrics.roc_auc_score(X["target"], pred)
    return auroc



if __name__ == "__main__":
    # conf = get_config()

    # prepare data
    # train_base_static = prepare_base_data(conf)

    # feature selection

    # model training

    # model evaluation

    # model prediction


    # test codes
    train_base_static = prepare_base_data()
    devval(train_base_static)
    dev = train_base_static[train_base_static["devval"] == 0]
    val = train_base_static[train_base_static["devval"] == 1]
    test = train_base_static[train_base_static["devval"] == 2]

    selector = get_tree_selector(
        exclude_object_columns(dev).fillna(-99999999), "target")

    dev_t = selector.transform(
        exclude_object_columns(dev)
        .fillna(-99999999)
        .drop("target", axis=1))
    val_t = selector.transform(
        exclude_object_columns(val)
        .fillna(-99999999)
        .drop("target", axis=1))
    test_t = selector.transform(
        exclude_object_columns(test)
        .fillna(-99999999)
        .drop("target", axis=1))    
    print(dev_t.shape, val_t.shape, test_t.shape)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, dev_t, dev["target"]),
        n_trials=50,
        show_progress_bar=True)

    print("Best trial:")
    trial = study.best_trial

    # retrain model with best params
    best_params = trial.params
    dtrain = lgb.Dataset(dev_t, label=dev["target"])
    best_model = lgb.train(best_params, dtrain)

    # inference
    dev_auroc = inference(selector, best_model, dev)
    print(f"dev auroc: {dev_auroc}")

    val_auroc = inference(selector, best_model, val)
    print(f"val auroc: {val_auroc}")

    test_auroc = inference(selector, best_model, test)
    print(f"test auroc: {test_auroc}")

    # inference monthly
    months = train_base_static["MONTH"].unique()
    auroc_list = []
    for month in months:
        month_df = train_base_static[train_base_static["MONTH"] == month]
        month_auroc = inference(selector, best_model, month_df)
        print(f"{month} auroc: {month_auroc}")
        auroc_list.append(month_auroc)

    # plot monthly auroc
    plt.plot(list(map(str, months)), auroc_list, "ro-")
