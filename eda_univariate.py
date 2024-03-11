import pandas as pd
from utils.eda_utils import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

################################################################################
#   train base (depth: 0, num_files: 1)
################################################################################
train_base = read_data("train_base", 0)
train_base.info()
display(train_base.head())
train_base["MONTH"] = train_base["MONTH"].astype(str)

monthly_agg = train_base.groupby("MONTH", dropna=False)["target"].agg(["count", "sum"]).sort_index()
monthly_agg["rate"] = monthly_agg["sum"] / monthly_agg["count"]
monthly_agg["count_pct"] = monthly_agg["count"] / monthly_agg["count"].sum()

# line for rate, bar for count_pct
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
monthly_agg["count_pct"].plot(ax=ax1, kind="bar", alpha=0.5)
monthly_agg["rate"].plot(ax=ax2, color="r", marker="o")
plt.show()

################################################################################
# fine classing
DROP_COLUMNS = [
    "dev_all_total",
    "dev_good_total",
    "dev_bad_total",
    "dev_bad_proportion",
    "dev_good_proportion",
    "dev_IV",
]
################################################################################
# 내부정보
FILE_NAME = "train_static"
DEPTH = 0
data = read_data(FILE_NAME, DEPTH)

depth0 = data.merge(
    train_base[["case_id", "target"]], on="case_id", how="left", suffixes=("", "_base")
)
df = fineclassing(depth0, depth0.columns, "target")
df["table"] = f"{FILE_NAME}_{DEPTH}"

df.drop(columns=DROP_COLUMNS).to_csv(Path("data") / "eda" / f"{FILE_NAME}_{DEPTH}.csv")

################################################################################
# 외부정보
FILE_NAME = "train_static"
DEPTH = "cb"
data = read_data(FILE_NAME, DEPTH)

depth0 = data.merge(
    train_base[["case_id", "target"]], on="case_id", how="left", suffixes=("", "_base")
)

df = fineclassing(depth0, depth0.columns, "target")
df["table"] = f"{FILE_NAME}_{DEPTH}"
df.drop(columns=DROP_COLUMNS).to_csv(Path("data") / "eda" / f"{FILE_NAME}_{DEPTH}.csv")

################################################################################
################################################################################
# 세금납부정보
data = read_data("train_tax_registry", 'a')
describe_data(data)
data.case_id.nunique()
# 0 case_id: case_id (postfix type : -) 28631
# 1 amount_4527230A: 정부 레지스트리에서 추적된 세액 공제액. (postfix type : Transform Amount) 1946.0
# 2 name_4527232M: 고용주의 이름. (postfix type : Masking Categories) f980a1ea
# 3 num_group1: num_group1 (postfix type : -) 2
# 4 recorddate_4527225D: 세금 공제 기록 날짜. (postfix type : Transform Date) 2019-09-13

# NUM_GROUP1 세금 납부 SEQ
# 사람에 따라서 여러개의 세금납부정보가 있음
data.groupby("case_id")["num_group1"].count().describe()
# count    457934.000000
# mean          7.153367
# std           3.818065
# min           1.000000
# 25%           5.000000
# 50%           6.000000
# 75%           8.000000
# max          99.000000
# Name: num_group1, dtype: float64

data.groupby("case_id")["recorddate_4527225D"].nunique().value_counts()
# 그러나 한 사람은 하나의 recorddate_4527225D만 가짐

data.groupby("case_id")["name_4527232M"].nunique().value_counts()
# 그러나 한 사람은 여러개의 recorddate_4527225D만 가짐
# '다닌 회사 수' 의 의미 도출
data["name_4527232M"].nunique()
# 데이터에는 고용주 이름이 147037개 있음 (이걸 키로 엮어?)

# feature 제안
# 세금납부정보의 개수
# amount_4527230A의 합, 평균
# 최근 n개의 amount_4527230A의 합, 평균
# name_4527232M의 개수, unique 개수

tax1 = data.copy()
# NUM_GROUP1 세금 납부 SEQ

################################################################################
data = read_data("train_tax_registry", 'b')
describe_data(data)
tax2 = data.copy()
# 예상
# NUM_GROUP1 세금 공제 SEQ
pd.DataFrame(tax1.case_id.unique(), columns=["case_id"]).merge(
    pd.DataFrame(tax2.case_id.unique(), columns=["case_id"]),
    on="case_id",
    how="inner",
).shape


################################################################################
data = read_data("train_tax_registry", 'c')
describe_data(data)
# 예상
# NUM_GROUP1 신청시 세금 공제 SEQ


################################################################################
################################################################################
# 이전 신청서정보
data = read_data("train_applprev", 1)
describe_data(data)
# 예상
# NUM_GROUP1 이전 신청서의 신청서 제출 SEQ

################################################################################
data = read_data("train_applprev", 2)
describe_data(data)

# 예상
# NUM_GROUP1 이전 신청서의 신청서 제출 SEQ
# NUM_GROUP2 이전 신청서의 신청인 SEQ


################################################################################
################################################################################
# 개인정보
data = read_data("train_person", 1)
describe_data(data)

# NUM_GROUP1 신청인 SEQ

################################################################################
data = read_data("train_person", 2)
describe_data(data)

# NUM_GROUP1 신청인 SEQ
# NUM_GROUP1 관련인 SEQ
#  In case num_group1 or num_group2 stands for person index (this is clear with predictor definitions)
#  the zero index has special meaning.
#  When num_groupN=0 it is the applicant (the person who applied for a loan)


################################################################################
################################################################################
# 직불카드정보
data = read_data("train_debitcard", 1)
describe_data(data)

# NUM_GROUP1 카드 SEQ?

################################################################################
################################################################################
# 예금정보
data = read_data("train_deposit", 1)
describe_data(data)

# NUM_GROUP1 신청인 SEQ? 계좌 SEQ?

################################################################################
################################################################################
# 기타정보
data = read_data("train_other", 1)
describe_data(data)

# NUM_GROUP1 모두 0 의미없음


################################################################################
################################################################################
# CB정보 A
data = read_data("train_credit_bureau_a", 1)
describe_data(data)

data = read_data("train_credit_bureau_a", 2, num_files=2)
describe_data(data)

################################################################################
# CB정보 B
data = read_data("train_credit_bureau_b", 1)
describe_data(data)

# CB정보 B
data = read_data("train_credit_bureau_b", 2)
describe_data(data)
