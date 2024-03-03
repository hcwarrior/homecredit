import os
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SelectFromModel
# import pandas as pd
from dataset.datainfo import RawInfo
from argparse import ArgumentParser, Namespace


def get_config():
    base_path = os.getcwd()
    data_path = os.path.join(base_path, "data/home-credit-credit-risk-model-stability")

    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default=base_path)
    parser.add_argument("--data_path", type=str, default=data_path)
    parser.add_argument("--raw_format", type=str, default="parquet")

    return parser.parse_args()


def prepare_data(conf: Namespace, type_: str = "train"):
    infos = RawInfo(conf)
    base_df = infos.read_raw("base", type_=type_)
    static_df = infos.read_raw("static", depth=0, type_=type_)

    return static_df.join(base_df[["case_id", "MONTH", "target"]], on="case_id", how="left", rsuffix="_base")

if __name__ == "__main__":
    conf = get_config()

    # prepare data
    train_base_static = prepare_data(conf)
    print(train_base_static.head())

    # feature selection

    # model training

    # model evaluation

    # model prediction


