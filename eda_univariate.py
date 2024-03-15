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
FILE_NAME = "train_tax_registry"
DEPTH = "a"
data = read_data(FILE_NAME, DEPTH)
tax1 = data.copy()
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
data.num_group1.value_counts()
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
# 한 CASE는 하나의 recorddate_4527225D만 가짐

data.groupby("case_id")["name_4527232M"].nunique().value_counts()
# 그러나 한 CASE은 여러개의 name_4527232M을 가짐
# '다닌 회사 수' 의 의미 도출
data["name_4527232M"].nunique()
# 데이터에는 고용주 이름이 147037개 있음
# TODO 이걸 키로 엮어?

# feature 제안
# 세금납부정보의 개수
# amount_4527230A의 합, 평균
# num_group1 = 1인 amount_4527230A의 합, 평균
# name_4527232M의 개수, unique 개수

data_agg = data.groupby("case_id").agg(
    num_group1=("num_group1", "count"),
    amount_4527230A_sum=("amount_4527230A", "sum"),
    amount_4527230A_mean=("amount_4527230A", "mean"),
    name_4527232M_nunique=("name_4527232M", "nunique"),
)
data_agg = data_agg.merge(
    data[data["num_group1"] == 0]
    .groupby("case_id")
    .agg(
        num1_0_amount_4527230A_mean=("amount_4527230A", "mean"),
    ),
    on="case_id",
    how="left",
)
data_agg = data_agg.merge(
    data[data["num_group1"] < 3]
    .groupby("case_id")
    .agg(
        num1_3_amount_4527230A_mean=("amount_4527230A", "mean"),
        num1_3_amount_4527230A_sum=("amount_4527230A", "sum"),
    ),
    on="case_id",
    how="left",
)

depth0 = data_agg.merge(
    train_base[["case_id", "target"]], on="case_id", how="left", suffixes=("", "_base")
)
depth0.corr()
df = fineclassing(depth0, depth0.columns, "target")
df["table"] = f"{FILE_NAME}_{DEPTH}"
df.drop(columns=DROP_COLUMNS).to_csv(Path("data") / "eda" / f"{FILE_NAME}_{DEPTH}.csv")

################################################################################
FILE_NAME = "train_tax_registry"
DEPTH = "b"
data = read_data(FILE_NAME, DEPTH)
describe_data(data)
tax2 = data.copy()
data.case_id.nunique()
# 0 case_id: case_id (postfix type : -) 49435
# 1 amount_4917619A: 정부 레지스트리에서 추적된 세액 공제액. (postfix type : Transform Amount) 6885.0
# 2 deductiondate_4917603D: 세액 공제 날짜. (postfix type : Transform Date) 2019-10-16
# 3 name_4917606M: 고용주의 이름. (postfix type : Masking Categories) 6b730375
# 4 num_group1: num_group1 (postfix type : -) 7

# 예상
# NUM_GROUP1 세금 공제 SEQ
pd.DataFrame(tax1.case_id.unique(), columns=["case_id"]).merge(
    pd.DataFrame(tax2.case_id.unique(), columns=["case_id"]),
    on="case_id",
    how="inner",
).shape
# INNER JOIN 결과 4246건


# NUM_GROUP1 세금 납부 SEQ
data.groupby("case_id")["num_group1"].count().describe()
# count    150732.000000
# mean          7.350350
# std           3.775325
# min           1.000000
# 25%           6.000000
# 50%           6.000000
# 75%           8.000000
# max         101.000000
# Name: num_group1, dtype: float64

# amount_4917619A
data.groupby("case_id")["deductiondate_4917603D"].nunique().value_counts()
# 아? 한 CASE는 여러개의 deductiondate_4917603D 가짐
# a의 recorddate_4527225D는 기록일이므로 한 case에 대한 정보수집일이라고 볼 수 있고
# b의 deductiondate_4917603D는 세금공제일이므로 한 case에 대한 과거의 세금공제 seq라고 볼 수 있음
data.groupby(["case_id", "num_group1"])["deductiondate_4917603D"].nunique().value_counts()
# 이건 당연한거고...

data.groupby("case_id")["name_4917606M"].nunique().value_counts()
# 그러나 한 CASE은 여러개의 name_4917606M 가짐
# '다닌 회사 수' 의 의미 도출
data["name_4917606M"].nunique()
# 데이터에는 고용주 이름이 55857개 있음 (이걸 키로 엮어?)

# feature 제안
# 세금납부정보의 개수
# 최근 00기간 내 세금납부정보의 개수
# amount_4917619A 합, 평균
# 최근 00기간 내 amount_4917619A 합, 평균
# name_4917606M 개수, unique 개수
# 최근 00기간 내 name_4917606M 개수, unique 개수

data[data["case_id"] == 49576].sort_values("deductiondate_4917603D")
data[data["case_id"] == 2703452].sort_values("deductiondate_4917603D")
data[data["case_id"] == 49576].sort_values("deductiondate_4917603D")
# seq와 날짜가 같이 증가하는 것으로 보아 seq가 시간순서대로 되어있음
# 날짜를 보았을 때 최근 6개월의 월급에서 공제된 세금을 의미하는 것으로 보임
# num_group1의 분포를 봤을 때 a도 최근 6개월 정보임을 추정할 수 있음
# 몇 종류의 세금을 내는지도 중요할 수 있음
# 월별 세금납부정보의 편차가 영향이 있을지도(급여생활자는 수입이 일정할테니까)
# 코로나 영향도 급여생활자와 자영업자가 다를것임
# 편차계산보다 distinct amount가 쉬울지도

data_agg = data.groupby("case_id").agg(
    num_group1=("num_group1", "count"),
    amount_4917619A_sum=("amount_4917619A", "sum"),
    amount_4917619A_nunique=("amount_4917619A", "nunique"),
    amount_4917619A_mean=("amount_4917619A", "mean"),
    name_4917606M_nunique=("name_4917606M", "nunique"),
    deductiondate_4917603D_nunique=("deductiondate_4917603D", "nunique"),
)
data_agg = data_agg.merge(
    data[data["num_group1"] == 0]
    .groupby("case_id")
    .agg(
        num1_0_amount_4527230A_mean=("amount_4917619A", "mean"),
    ),
    on="case_id",
    how="left",
)
data_agg = data_agg.merge(
    data[data["num_group1"] < 3]
    .groupby("case_id")
    .agg(
        num1_3_amount_4527230A_mean=("amount_4917619A", "mean"),
        num1_3_amount_4527230A_sum=("amount_4917619A", "sum"),
    ),
    on="case_id",
    how="left",
)
data_agg = data_agg.merge(
    data.groupby(["case_id", "deductiondate_4917603D"])
    .agg(
        amount_4917619A_sum=("amount_4917619A", "sum"),
    ).groupby("case_id")
    .agg(
        amount_4917619A_sum_std=("amount_4917619A_sum", "std"),
    ),
    on="case_id",
    how="left",
)

depth0 = data_agg.merge(
    train_base[["case_id", "target"]], on="case_id", how="left", suffixes=("", "_base")
)
depth0.corr()
df = fineclassing(depth0, depth0.columns, "target")
df["table"] = f"{FILE_NAME}_{DEPTH}"
df.drop(columns=DROP_COLUMNS).to_csv(Path("data") / "eda" / f"{FILE_NAME}_{DEPTH}.csv")

################################################################################
FILE_NAME = "train_tax_registry"
DEPTH = "c"
data = read_data(FILE_NAME, DEPTH)
describe_data(data)
data.case_id.nunique()
# 0 case_id: case_id (postfix type : -) 357
# 1 employername_160M: 고용주의 이름. (postfix type : Masking Categories) c91b12ff
# 2 num_group1: num_group1 (postfix type : -) 5
# 3 pmtamount_36A: 신용 레포트 지불을 위한 세금 공제 금액. (postfix type : Transform Amount) 1100.0
# 4 processingdate_168D: 세금 공제가 처리된 날짜. (postfix type : Transform Date) 2018-08-08


data.groupby("case_id")["processingdate_168D"].nunique().value_counts()
# b와 같은 패턴
data.groupby("case_id")["employername_160M"].nunique().value_counts()
data["employername_160M"].nunique()
# b와 같은 패턴

# 예상
# NUM_GROUP1 신청시 세금 공제 SEQ

# feature 제안
# b와 같은 패턴
data_agg = data.groupby("case_id").agg(
    num_group1=("num_group1", "count"),
    pmtamount_36A_sum=("pmtamount_36A", "sum"),
    pmtamount_36A_nunique=("pmtamount_36A", "nunique"),
    pmtamount_36A_mean=("pmtamount_36A", "mean"),
    employername_160M_nunique=("employername_160M", "nunique"),
    processingdate_168D_nunique=("processingdate_168D", "nunique"),
)
data_agg = data_agg.merge(
    data[data["num_group1"] == 0]
    .groupby("case_id")
    .agg(
        num1_0_pmtamount_36A_mean=("pmtamount_36A", "mean"),
    ),
    on="case_id",
    how="left",
)
data_agg = data_agg.merge(
    data[data["num_group1"] < 3]
    .groupby("case_id")
    .agg(
        num1_3_pmtamount_36A_mean=("pmtamount_36A", "mean"),
        num1_3_pmtamount_36A_sum=("pmtamount_36A", "sum"),
    ),
    on="case_id",
    how="left",
)
data_agg = data_agg.merge(
    data.groupby(["case_id", "processingdate_168D"])
    .agg(
        pmtamount_36A_sum=("pmtamount_36A", "sum"),
    ).groupby("case_id")
    .agg(
        pmtamount_36A_sum_std=("pmtamount_36A_sum", "std"),
    ),
    on="case_id",
    how="left",
)
# TODO: a, b, c 를 합산해서 만든 세금납부액정보를 만들어야함

depth0 = data_agg.merge(
    train_base[["case_id", "target"]], on="case_id", how="left", suffixes=("", "_base")
)
depth0.corr()
df = fineclassing(depth0, depth0.columns, "target")
df["table"] = f"{FILE_NAME}_{DEPTH}"
df.drop(columns=DROP_COLUMNS).to_csv(Path("data") / "eda" / f"{FILE_NAME}_{DEPTH}.csv")


################################################################################
################################################################################
# 이전 신청서정보
FILE_NAME = "train_applprev"
DEPTH = 1
data = read_data(FILE_NAME, DEPTH)
describe_data(data)
# 0 case_id: case_id (postfix type : -) 2
# 31 num_group1: num_group1 (postfix type : -) 0

# 3 approvaldate_319D: 이전 신청서의 승인 날짜. (postfix type : Transform Date) 2019-01-11
# 7 creationdate_885D: 이전 신청서가 작성된 날짜. (postfix type : Transform Date) 2013-04-03
# 17 dateactivated_425D: 이전 신청서의 계약 활성화 날짜. (postfix type : Transform Date) 2018-10-19
# 20 dtlastpmt_581D: 신청자의 마지막 지불 날짜. (postfix type : Transform Date) 2019-01-10
# 21 dtlastpmtallstes_3545839D: 신청인의 마지막 지불 날짜. (postfix type : Transform Date) 2019-01-10
# 23 employedfrom_700D: 이전 신청서에서의 고용 시작일. (postfix type : Transform Date) 2010-02-15
# 25 firstnonzeroinstldate_307D: 이전 신청서에서의 첫 번째 할부 날짜. (postfix type : Transform Date) 2013-05-04

# 18 district_544M: 이전 대출 신청서에서 사용된 주소의 구역. (postfix type : Masking Categories) P136_108_173
# 22 education_1138M: 이전 신청서의 신청인의 교육 수준. (postfix type : Masking Categories) P97_36_170
# 5 cancelreason_3545846M: 신청서 취소 이유. (postfix type : Masking Categories) a55475b1
# 34 postype_4733339M: 점포 유형. (postfix type : Masking Categories) a55475b1
# 35 profession_152M: 고객의 이전 대출 신청 시 직업. (postfix type : Masking Categories) a55475b1
# 36 rejectreason_755M: 이전 신청 거부 사유. (postfix type : Masking Categories) a55475b1
# 37 rejectreasonclient_4145042M: 고객의 이전 신청 거부 사유. (postfix type : Masking Categories) a55475b1
# 12 credacc_status_367L: 이전 신청서의 계좌 상태. (postfix type : Unspecified Transform) CL
# 26 inittransactioncode_279L: 신청인의 이전 신청서의 초기 거래 유형. (postfix type : Unspecified Transform) CASH
# 39 status_219L: 이전 신청 상태. (postfix type : Unspecified Transform) D
# 15 credtype_587L: 이전 신청서의 신용 유형. (postfix type : Unspecified Transform) CAL
# 24 familystate_726L: 신청인의 이전 신청서의 가족 상태. (postfix type : Unspecified Transform) SINGLE

# 27 isbidproduct_390L: 이전 신청에서 제품이 교차 판매인지를 결정하는 플래그. (postfix type : Unspecified Transform) False
# 28 isdebitcard_527L: 신청 중인 제품이 직불 카드인지를 나타내는 이전 신청 플래그. (postfix type : Unspecified Transform) False

# 1 actualdpd_943P: 이전 계약의 지연일수(DPD). (postfix type : Transform DPD (Days Past Due)) 0.0
# 2 annuity_853A: 이전 신청서의 월 상환액. (postfix type : Transform Amount) 640.2
# 4 byoccupationinc_3656910L: 이전 신청서의 신청인 소득. (postfix type : Unspecified Transform) 1.0
# 6 childnum_21L: 이전 신청에서의 자녀 수. (postfix type : Unspecified Transform) 0.0
# 8 credacc_actualbalance_314A: 신용 계좌의 실질 잔액. (postfix type : Transform Amount) 30450.0
# 9 credacc_credlmt_575A: 이전 신청서에 제공된 신용 카드 신용 한도. (postfix type : Transform Amount) 0.0
# 10 credacc_maxhisbal_375A: 이전 신청서의 최대 신용 카드 잔액 (postfix type : Transform Amount) 0.0
# 11 credacc_minhisbal_90A: 이전 신용 계좌의 최소 잔액. (postfix type : Transform Amount) 0.0
# 13 credacc_transactions_402L: 신청자의 이전 신용 계좌에서의 거래 횟수. (postfix type : Unspecified Transform) 0.0
# 14 credamount_590A: 이전 신청서의 대출 금액 또는 신용 카드 한도. (postfix type : Transform Amount) 10000.0
# 16 currdebt_94A: 이전 신청서의 현재 부채. (postfix type : Transform Amount) 0.0
# 19 downpmt_134A: 이전 신청서의 선수금액. (postfix type : Transform Amount) 0.0
# 29 mainoccupationinc_437A: 고객의 이전 신청에서의 주요 소득 금액. (postfix type : Transform Amount) 8200.0
# 30 maxdpdtolerance_577P: 이전 신청에서의 최대 DPD(연체 일수). (postfix type : Transform DPD (Days Past Due)) 0.0
# 32 outstandingdebt_522A: 고객의 이전 신청에 대한 미지급 부채액. (postfix type : Transform Amount) 0.0
# 33 pmtnum_8L: 이전 신청에 대한 지불 횟수. (postfix type : Unspecified Transform) 24.0
# 38 revolvingaccount_394A: 신청자의 이전 신청에서 존재한 리볼빙 계정. (postfix type : Transform Amount) 760645950.0
# 40 tenor_203L: 이전 신청서의 할부 개수. (postfix type : Unspecified Transform) 24.0

data.approvaldate_319D.str[:7].value_counts().sort_index()
# 승인날짜로 승인여부 확인 가능
data[
    [
        "creationdate_885D",
        "approvaldate_319D",
        "dateactivated_425D",
        "firstnonzeroinstldate_307D",
        "dtlastpmt_581D",
        "dtlastpmtallstes_3545839D",
    ]
].iloc[-50:, :]
# case_id와 num_group1을 key로 해서 label을 붙여서 평가해보면 어떨까?햐

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
