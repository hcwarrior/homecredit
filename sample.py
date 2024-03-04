import os
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SelectFromModel
# import pandas as pd
from dataset.datainfo import RawInfo
from argparse import ArgumentParser, Namespace
import pandas as pd


def get_config():
    base_path = os.getcwd()
    data_path = os.path.join(base_path, "data/home-credit-credit-risk-model-stability")

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

if __name__ == "__main__":
    conf = get_config()

    # prepare data
    # train_base_static = prepare_data(conf)
    train_base_static = prepare_base_data()

    # feature selection

    # model training

    # model evaluation

    # model prediction

