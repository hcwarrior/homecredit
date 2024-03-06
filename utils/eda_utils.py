from typing import Union
from pandas.api.types import is_numeric_dtype
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import glob
import decimal
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from IPython.display import display


BASE_PATH = Path(os.getcwd())
DATA_PATH = BASE_PATH / "data" / "home-credit-credit-risk-model-stability"
PARQUET_DIR_PATH = DATA_PATH / "parquet_files"
DESCRIPTION_DF = pd.read_csv(DATA_PATH / "feature_definitions.csv")

def describe(col: str, description_df: pd.DataFrame = DESCRIPTION_DF):
    if col in description_df["Variable"].values:
        result = description_df.loc[description_df["Variable"] == col, "Description_kor"].values[0]
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

    return f"{col}: {result} (postfix type : {postfix_dict.get(col[-1], '-')})"

def describe_data(data: pd.DataFrame):
    data.info()
    display(data.head())
    for i, c in enumerate(data.columns):
        print(i, describe(c), data[c].dropna().iloc[0])
    return data.describe().astype(int)

def read_data(
    file_name: str,
    depth: int,
    gubun: str = "train",
    num_files: int = 1,
    dir_path: str = PARQUET_DIR_PATH,
    format_: str = "parquet",
):
    file_dir = os.path.join(dir_path, gubun)
    file_name_format = f"{file_name}_{depth}*.{format_}"

    if file_name in ["train_base", "test_base"]:
        file_path = os.path.join(file_dir, f"{file_name}.{format_}")
        return pd.read_parquet(file_path)

    files = [f for f in glob.glob(os.path.join(file_dir, file_name_format))]
    if len(files) == 0:
        raise FileNotFoundError(
            f"No file found with the name '{file_name_format}' in '{file_dir}'"
        )

    return pd.read_parquet(files)


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

    if is_numeric_dtype(df_[col]):
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

    if is_numeric_dtype(df_[col]):
        df_rslt = df_rslt.sort_index(key=lambda x: x.map(lambda x: extract_first_number(x, special_codes)))
    else:
        df_rslt = df_rslt.sort_index()
    
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
        x_axis_str = final_df.index.map(str)
        x_axis_str = list(map(str, final_df.index.to_numpy()))
        fig, ax1 = plt.subplots()
        ax1.bar(x_axis_str, final_df["Count (%)"].to_numpy())
        ax2 = ax1.twinx()
        ax2.plot(x_axis_str, final_df["Event rate"].to_numpy(), "ro-")
        fig.tight_layout()
        plt.show()

    return final_df, sorted_splits


def is_number(s):
    try:
        decimal.Decimal(s)
        return True
    except decimal.InvalidOperation:
        return False
    except TypeError:
        return False


def classing_each(col, data, y, val_gb, ignore_sv):

    if data.dtype in ("object", "O") and data.apply(is_number).all():
        data = data.astype("double")
    sv_list = [-99999999, np.NaN, "Missing", "Special", np.inf, -np.inf]

    if ignore_sv:
        for each in (y, val_gb, data):
            each = each[~data.isin(sv_list)]

    if len(data) == 0:
        return

    if data.dtype in ("object", "O", "category") or data.nunique() <= 5:
        binned = data
        report = (
            pd.DataFrame({"col": col, "bin": binned, "label": y, "val_gb": val_gb})
            .value_counts()
            .reset_index()
        )
        report["order"] = report["bin"]
    else:
        binned = pd.qcut(data, 10, duplicates="drop")
        if binned.nunique() <= 1:
            binned = pd.qcut(data, 200, duplicates="drop")
        report = (
            pd.DataFrame({"col": col, "bin": binned, "label": y, "val_gb": val_gb})
            .value_counts()
            .reset_index()
        )
        report["order"] = report["bin"].cat.codes
    report = report.sort_values(["order", "label"])
    report = report.drop(columns=["order"]).set_index(["col", "bin"])

    return report


def fineclassing(data, use_columns, label="label", val_gb_colname=None, ignore_sv=False)-> pd.DataFrame:
    '''Get reports for each column in the data.
    Args:
        data: DataFrame
        use_columns: list of str
        label: str
        val_gb_colname: str
        ignore_sv: bool
    Returns:
        DataFrame
        '''
    y = data[label]
    val_gb = data[val_gb_colname] if val_gb_colname else 0

    with Parallel(n_jobs=-1) as parallel:
        reports = parallel(
            delayed(classing_each)(col, data[col], y, val_gb, ignore_sv)
            for col in tqdm(use_columns)
        )
    print(reports)

    reports = pd.concat(reports)
    reports = pd.concat(
        [
            reports[(reports["label"] == 0) & (reports["val_gb"] == 0)].iloc[:, 2],
            reports[(reports["label"] == 1) & (reports["val_gb"] == 0)].iloc[:, 2],
            reports[(reports["label"] == 0) & (reports["val_gb"] == 1)].iloc[:, 2],
            reports[(reports["label"] == 1) & (reports["val_gb"] == 1)].iloc[:, 2],
        ],
        axis=1,
    )
    reports.columns = ["dev_good", "dev_bad", "val_good", "val_bad"]
    reports = reports.fillna(0)
    # calc bad_rate
    reports["dev_all"] = reports["dev_good"] + reports["dev_bad"]
    reports["dev_bad_rate"] = reports["dev_bad"] / reports["dev_all"]
    reports["val_all"] = reports["val_good"] + reports["val_bad"]
    reports["val_bad_rate"] = reports["val_bad"] / reports["val_all"]
    # calc total
    reports.reset_index().set_index(["col"])
    reports = reports.join(
        reports.groupby(level=0).sum()[
            ["dev_all", "dev_good", "dev_bad", "val_all", "val_good", "val_bad"]
        ],
        rsuffix="_total",
    )
    # calc IV
    reports["dev_bad_proportion"] = reports["dev_bad"] / reports["dev_bad_total"]
    reports["dev_good_proportion"] = reports["dev_good"] / reports["dev_good_total"]
    reports["dev_all_proportion"] = reports["dev_all"] / reports["dev_all_total"]
    reports["dev_IV"] = np.where(
        (reports["dev_good_proportion"] == 0) | (reports["dev_bad_proportion"] == 0),
        0,
        (reports["dev_good_proportion"] - reports["dev_bad_proportion"])
        * np.log(reports["dev_good_proportion"] / reports["dev_bad_proportion"]),
    )
    reports["val_bad_proportion"] = reports["val_bad"] / reports["val_bad_total"]
    reports["val_good_proportion"] = reports["val_good"] / reports["val_good_total"]
    reports["val_all_proportion"] = reports["val_all"] / reports["val_all_total"]
    reports["val_IV"] = np.where(
        (reports["val_good_proportion"] == 0) | (reports["val_bad_proportion"] == 0),
        0,
        (reports["val_good_proportion"] - reports["val_bad_proportion"])
        * np.log(reports["val_good_proportion"] / reports["val_bad_proportion"]),
    )
    reports = reports.replace([np.inf, -np.inf], np.nan)
    reports = reports.join(
        reports.groupby(level=0).sum()[["dev_IV", "val_IV"]], rsuffix="_total"
    )
    # calc PSI
    reports["PSI"] = (
        reports["val_all_proportion"] - reports["dev_all_proportion"]
    ) * np.log(reports["val_all_proportion"] / reports["dev_all_proportion"])
    reports = reports.join(reports.groupby(level=0).sum()[["PSI"]], rsuffix="_total")
    # summary
    reports.reset_index().set_index(["col", "bin"])

    if val_gb_colname is None:
        reports = reports[[c for c in reports.columns if "dev" in c]]

    return reports
