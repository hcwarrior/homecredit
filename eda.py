import pandas as pd
from utils.eda_utils import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

description_df = pd.read_csv(f"{DATA_PATH}/feature_definitions.csv")


################################################################################
#   train base (depth: 0, num_files: 1)
################################################################################
train_base = read_data("train_base", 0, num_files=1)
train_base.info()
train_base.head()

monthly_agg = train_base.groupby("MONTH", dropna=False)["target"].agg(["count", "sum"]).sort_index()
monthly_agg["rate"] = monthly_agg["sum"] / monthly_agg["count"]
monthly_agg["count_pct"] = monthly_agg["count"] / monthly_agg["count"].sum()

# monthly_agg.plot(kind="bar", y="count_pct", figsize=(10, 5))
# plt.plot(monthly_agg.index, monthly_agg["rate"], "ro-")
# plt.show()



################################################################################
#   train static (depth: 0, num_files: 2)
################################################################################

train_static = read_data("train_static", 0, num_files=2)
train_static.info()
train_static.head()

train_depth0 = train_static.join(train_base[["case_id", "target"]], on="case_id", how="left", rsuffix="_base")

col = train_depth0.columns[5]
for idx, col in enumerate(train_depth0.columns):
    print(idx+1, "/", len(train_depth0.columns))
    print(describe(col))
    d, s = finebinning(
        train_depth0,
        col,
        "target",
        20,
        True,
        1,
    )

(train_depth0["actualdpdtolerance_344P"]>0).value_counts()

################################################################################
#   train static cb (0)
################################################################################

# train_static_cb = pd.read_parquet(f"{PARQUET_DIR_PATH}/train/train_static_cb_0.parquet")
# train_static_cb.info()
# train_static_cb.head()
# train_static.groupby(["case_id"], dropna=False)["actualdpdtolerance_344P"].nunique().value_counts()




# train_applprev_1_0 = pd.read_parquet(f"{PARQUET_DIR_PATH}/train/train_applprev_1_0.parquet")
# train_applprev_1_0.info()
# train_applprev_1_1 = pd.read_parquet(f"{PARQUET_DIR_PATH}/train/train_applprev_1_1.parquet")
# train_applprev_1_1.info()
# train_applprev_2 = pd.read_parquet(f"{PARQUET_DIR_PATH}/train/train_applprev_2.parquet")
# train_applprev_2.info()

# train_applprev_2.columns
# {col: describe(col) for col in train_applprev_2.columns}

# train_applprev_1_0.head()
# [c for c in train_applprev_1_0.columns if 'group' in c]
# train_applprev_1_0.columns.to_list().index('num_group1')

# describe("actualdpd_943P")


# train_applprev_1_0[["case_id","num_group1"]].duplicated().sum()
# train_applprev_1_0[["case_id"]].duplicated().sum()
# train_applprev_1_0[["case_id"]].duplicated().sum()
# train_applprev_1_0.shape

# describe("pmtnum_8L")
# describe("cacccardblochreas_147M")
# describe("conts_type_509L")
# describe("credacc_cards_status_52L")
# describe("actualdpdtolerance_344P")



# train_applprev_2.groupby(["case_id","num_group1"])["num_group2"].nunique().value_counts()
# train_applprev_2.groupby("case_id")["num_group1"].nunique().value_counts()
# train_applprev_2[["case_id","num_group1","num_group2"]].duplicated().sum()
# train_applprev_2.head(20)

# train_applprev_2[train_applprev_2.case_id == 3][["case_id", "num_group1", "num_group2"]]
# train_applprev_1_0[train_applprev_1_0.case_id == 3]