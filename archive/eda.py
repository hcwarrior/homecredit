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
################################################################################
# 내부정보
FILE_NAME = "train_static"
DEPTH = 0
data = read_data(FILE_NAME, DEPTH)
describe_data(data)

depth0 = data.merge(
    train_base[["case_id", "target"]], on="case_id", how="left", suffixes=("", "_base")
)
# df = fineclassing(depth0, depth0.columns, "target")

################################################################################
# 외부정보
FILE_NAME = "train_static"
DEPTH = "cb"
data = read_data(FILE_NAME, DEPTH)
# describe_data(data)

depth0 = data.merge(
    train_base[["case_id", "target"]], on="case_id", how="left", suffixes=("", "_base")
)

df = fineclassing(depth0, depth0.columns, "target")
display(df)

################################################################################
################################################################################
# 세금납부정보
data = read_data("train_tax_registry", 'a')
describe_data(data)
# 예상
# NUM_GROUP1 세금 납부 SEQ

################################################################################
data = read_data("train_tax_registry", 'b')
describe_data(data)
# 예상
# NUM_GROUP1 세금 공제 SEQ

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
