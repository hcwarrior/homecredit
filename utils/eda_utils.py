from typing import Union
from pandas.api.types import is_string_dtype
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


BASE_PATH = os.getcwd()
DATA_PATH = os.path.join(BASE_PATH, "data")
PARQUET_DIR_PATH = os.path.join(DATA_PATH, "parquet_files")

description_df = pd.read_csv(f"{DATA_PATH}/feature_definitions.csv")


def describe(col: str, description_df: pd.DataFrame = description_df):
    if col in description_df["Variable"].values:
        result = description_df.loc[description_df["Variable"] == col, "Description"].values[0]
    else:
        result = col
    
    postfix_dict = {
        "P": "Transform DPD (Days Past Due)",
        "M": "Masking Categories",
        "A": "Transform Amount",
        "D": "Transform Date",
        "T": "Unspecified Transform",
        "L": "Unspecified Transform",
    }

    if col[-1] in postfix_dict:
        return f"{col}: {result} ({postfix_dict[col[-1]]})"
    else:
        return f"{col}: result"


def read_data(
    file_name: str,
    depth: int,
    gubun: str = "train",
    num_files: int = 1,
    dir_path: str = PARQUET_DIR_PATH,
    format_: str = "parquet",
):
    file_dir = os.path.join(dir_path, gubun)
    file_name_format = "{file_name}_{depth}{postfix}.{format_}"
    
    if file_name in ["train_base", "test_base"]:
        file_path = os.path.join(file_dir, f"{file_name}.{format_}")
        return pd.read_parquet(file_path)

    if num_files > 1:
        df_ = pd.concat([
            pd.read_parquet(
                os.path.join(
                    file_dir,
                    file_name_format.format(
                        file_name=file_name,
                        depth=depth,
                        postfix=f"_{i}",
                        format_=format_))
            )
            for i in range(num_files)
        ])
    elif num_files == 1:
        file_path = os.path.join(
            file_dir,
            file_name_format.format(file_name=file_name, depth=depth, postfix="", format_=format_))
        df_ = pd.read_parquet(file_path)
    else:
        raise ValueError(f"num_files should be greater than 0. Not {num_files}.")

    return df_


def extract_first_number(x: str, special_codes: dict = None, large_number=9.9999e10) -> int:
    if x.startswith("(-inf"):
        return float("-inf")
    elif x == "Missing":
        return large_number
    elif x == "Special":
        return large_number - 1
    elif special_codes is not None and x in special_codes:
        return large_number - 1 - (len(special_codes) - list(special_codes.keys()).index(x))
    else:
        try:
            return float(x.split(",")[0][1:])
        except:
            return x


def get_labels_from_split(splits: list):
    return [f"[{left}, {right})" for left, right in zip(splits[:-1], splits[1:])]


def agg_basic_stats(df, col, tgt):
    df_rslt = df[col].value_counts().to_frame(name="Count")
    df_rslt["Event"] = df.groupby(col)[tgt].sum()
    df_rslt.index.names = ["Bins"]
    return df_rslt


def finebinning(
    df: pd.DataFrame,
    col: str,
    tgt: str,
    nbins: int = 25,
    plot: bool = False,
    target_value: Union[int, float, str] = 1,
    precision: int = 4,
    special_codes: dict = None,
    casting_type: str = None,
    pre_splits: list = None,
):
    df_ = df[[col, tgt]].copy()

    tgt_yn_str = "TARGET_YN_COLUMN"
    df_[tgt_yn_str] = df_[tgt] == target_value
    is_na = df_[col].isna()

    if casting_type is not None:
        df_[col] = df_[col].astype(casting_type)

    if special_codes is not None:
        sv_list = list(special_codes.values())
        is_sv = df_[col].isin(sv_list)
    else:
        is_sv = pd.Series([False] * len(df_))

    if not is_string_dtype(df_[col]):
        if pre_splits is not None:
            sorted_splits = pre_splits
        else:
            split_percentile = [round(x / nbins, 2) for x in range(nbins)] + [1]
            splits = df_.loc[~is_sv, col].quantile(split_percentile, interpolation="nearest")
            splits = np.round(splits, precision).drop_duplicates()
            sorted_splits = sorted(list(set(splits.values)))
            sorted_splits.insert(0, -np.inf)
            sorted_splits.append(np.inf)
        labels = get_labels_from_split(sorted_splits)
        df_.loc[~is_sv, col] = pd.cut(df_.loc[~is_sv, col], bins=sorted_splits, labels=labels, right=False)
    else:
        sorted_splits = []

    df_rslt = agg_basic_stats(df_.loc[~is_sv], col, tgt_yn_str)
    result_list = [df_rslt]

    if special_codes is not None:
        result_list.extend(
            pd.DataFrame(
                {"Count": df_[col].eq(spc_cd).sum(), "Event": df_.loc[df_[col].eq(spc_cd), tgt_yn_str].sum()},
                index=[spc_name],
            )
            for spc_name, spc_cd in special_codes.items()
        )

    result_list.append(pd.DataFrame({"Count": is_na.sum(), "Event": df_.loc[is_na, tgt_yn_str].sum()}, index=["Missing"]))

    final_df = pd.concat(result_list, axis=0)
    final_df = final_df[final_df["Count"] > 0]

    total_count = final_df["Count"].sum()
    event_count = final_df["Event"].sum()
    nonevent_count = total_count - event_count

    final_df["Non-event"] = final_df["Count"] - final_df["Event"]
    final_df["Count (%)"] = final_df["Count"] / total_count
    final_df["Event rate"] = final_df["Event"] / final_df["Count"]
    final_df["Event (%)"] = final_df["Event"] / event_count
    final_df["Non-event (%)"] = final_df["Non-event"] / nonevent_count
    final_df["IV"] = np.where(
        (final_df["Non-event (%)"] == 0) | (final_df["Event (%)"] == 0),
        0,
        (final_df["Event (%)"] - final_df["Non-event (%)"]) * np.log(final_df["Event (%)"] / final_df["Non-event (%)"]),
    )

    if plot:
        final_df.plot(kind="bar", y="Count (%)")
        plt.plot(final_df.index, final_df["Event rate"], "ro-")
        plt.show()

    return final_df.sort_index(key=lambda x: x.map(lambda x: extract_first_number(x, special_codes))), sorted_splits
