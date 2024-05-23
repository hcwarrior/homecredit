import gc
import os
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import polars as pl

from simple_parsing import ArgumentParser
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class Options:
    root_dir: str
    output_root_dir: str
    train_ratio: float
    output_root_dir: str
    train_split: int


class Pipeline:
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  # !!?
                df = df.with_columns(pl.col(col).dt.total_days())  # t - t-1
        df = df.drop("date_decision", "MONTH")
        return df

    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.7:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df


import numpy as np


class Pipeline:
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  # !!?
                df = df.with_columns(pl.col(col).dt.total_days())  # t - t-1
        df = df.drop("date_decision", "MONTH")
        return df

    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.7:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df


class Aggregator:
    # Please add or subtract features yourself, be aware that too many features will take up too much space.
    def num_expr(df, columns=None):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        return (
            []
            + Aggregator.get_last_cols(df, cols, columns)
            + Aggregator.get_max_cols(df, cols, columns)
            + Aggregator.get_mean_cols(df, cols, columns)
            + Aggregator.get_median_cols(df, cols, columns)
            + Aggregator.get_variance_cols(df, cols, columns)
        )


    def date_expr(df, columns=None):
        cols = [col for col in df.columns if col[-1] in ("D")]
        return (
                []
                + Aggregator.get_last_cols(df, cols, columns)
                + Aggregator.get_max_cols(df, cols, columns)
                + Aggregator.get_median_cols(df, cols, columns)
        )

    def str_expr(df, columns=None):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        return (
                []
                + Aggregator.get_last_cols(df, cols, columns)
                + Aggregator.get_max_cols(df, cols, columns)
                + Aggregator.get_mode_cols(df, cols, columns)
        )

    def other_expr(df, columns=None):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        return (
                []
                + Aggregator.get_last_cols(df, cols, columns)
                + Aggregator.get_max_cols(df, cols, columns)
                + Aggregator.get_min_cols(df, cols, columns)
        )

    def count_expr(df, columns=None):
        cols = [col for col in df.columns if "num_group" in col]
        return (
                []
                + Aggregator.get_last_cols(df, cols, columns)
                + Aggregator.get_max_cols(df, cols, columns)
                + Aggregator.get_min_cols(df, cols, columns)
        )

    def filter_columns(dfs, columns):
        if columns is None or len(columns) == 0:
            return dfs
        return [df for df in dfs if
                df.meta.output_name() in columns or any(col.startswith(df.meta.output_name()) for col in columns)]

    def get_exprs(df, columns):
        exprs = Aggregator.num_expr(df, columns) + \
                Aggregator.date_expr(df, columns) + \
                Aggregator.str_expr(df, columns) + \
                Aggregator.other_expr(df, columns) + \
                Aggregator.count_expr(df, columns)

        return exprs

    def get_max_cols(df, target_cols, cols):
        return Aggregator.filter_columns([pl.max(col).alias(f"max_{col}") for col in target_cols], cols)

    def get_min_cols(df, target_cols, cols):
        return Aggregator.filter_columns([pl.min(col).alias(f"min_{col}") for col in target_cols], cols)

    def get_last_cols(df, target_cols, cols):
        return Aggregator.filter_columns([pl.last(col).alias(f"last_{col}") for col in target_cols], cols)

    def get_sum_cols(df, target_cols, cols):
        return Aggregator.filter_columns([pl.sum(col).alias(f"sum_{col}") for col in target_cols], cols)

    def get_count_cols(df, target_cols, cols):
        return Aggregator.filter_columns([pl.count(col).alias(f"count_{col}") for col in target_cols], cols)

    def get_first_cols(df, target_cols, cols):
        return Aggregator.filter_columns([pl.first(col).alias(f"first_{col}") for col in target_cols], cols)

    def get_mean_cols(df, target_cols, cols):
        return Aggregator.filter_columns([pl.mean(col).alias(f"mean_{col}") for col in target_cols], cols)

    def get_mode_cols(df, target_cols, cols):
        return Aggregator.filter_columns([pl.col(col).mode().first().alias(f"mode_{col}") for col in target_cols], cols)

    def get_median_cols(df, target_cols, cols):
        return Aggregator.filter_columns([pl.median(col).alias(f"median{col}") for col in target_cols], cols)

    def get_variance_cols(df, target_cols, cols):
        return Aggregator.filter_columns([pl.var(col).alias(f"var{col}") for col in target_cols], cols)


def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df, None))
    return df


def read_files(regex_path, depth=None):
    chunks = []

    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df, None))
        chunks.append(df)

    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    return df


def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = (
        df_base
        .with_columns(
            month_decision = pl.col("date_decision").dt.month(),
            weekday_decision = pl.col("date_decision").dt.weekday(),
        )
    )
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base


def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) == "category":
            continue

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")

    args = parser.parse_args()
    options = args.options

    root_dir = Path(options.root_dir)
    data_store = {
        "df_base": read_file(root_dir / "train_base.parquet"),
        "depth_0": [
            read_file(root_dir / "train_static_cb_0.parquet"),
            read_files(root_dir / "train_static_0_*.parquet"),
        ],
        "depth_1": [
            read_files(root_dir / "train_applprev_1_*.parquet", 1),
            read_file(root_dir / "train_tax_registry_a_1.parquet", 1),
            read_file(root_dir / "train_tax_registry_b_1.parquet", 1),
            read_file(root_dir / "train_tax_registry_c_1.parquet", 1),
            read_files(root_dir / "train_credit_bureau_a_1_*.parquet", 1),
            read_file(root_dir / "train_credit_bureau_b_1.parquet", 1),
            read_file(root_dir / "train_other_1.parquet", 1),
            read_file(root_dir / "train_person_1.parquet", 1),
            read_file(root_dir / "train_deposit_1.parquet", 1),
            read_file(root_dir / "train_debitcard_1.parquet", 1),
        ],
        "depth_2": [
            read_file(root_dir / "train_credit_bureau_b_2.parquet", 2),
            read_files(root_dir / "train_credit_bureau_a_2_*.parquet", 2),
        ]
    }

    df = feature_eng(**data_store)
    df, _ = to_pandas(df)

    train_ratio, test_ratio = options.train_ratio, 1 - options.train_ratio

    df_train = df.sample(frac=train_ratio, random_state=42)
    df_test = df.drop(df_train.index)

    del df
    gc.collect()

    output_root_dir = Path(options.output_root_dir)

    train_split = options.train_split
    cv = StratifiedGroupKFold(n_splits=train_split, shuffle=False)

    i = 0
    for idx_train, idx_valid in cv.split(df_train, np.zeros(len(df_train)), groups=df_train['WEEK_NUM']):
        train_kfold = df_train.iloc[idx_train]
        val_kfold = df_train.iloc[idx_valid]

        os.makedirs(output_root_dir / f"train{i}", exist_ok=True)
        os.makedirs(output_root_dir / f"val{i}", exist_ok=True)

        train_kfold.to_parquet(output_root_dir / f"train{i}/data.parquet")
        val_kfold.to_parquet(output_root_dir / f"val{i}/data.parquet")

        i += 1

    os.makedirs(output_root_dir / "test", exist_ok=True)
    df_test.to_parquet(output_root_dir / "test/data.parquet")
