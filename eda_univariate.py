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

data[data["case_id"] == 2701515]

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
appl_1 = data.copy()
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

data.groupby("case_id")["num_group1"].count().describe().astype(int)
data["num_group1"].value_counts()

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
# case_id와 num_group1을 key로 해서 label을 붙여서 평가해보면 어떨까?

# 범주형 변수
# a55475b1를 공통으로 가장 많이 가지고 있는데, null같은 것으로 취급해도 될까?
# Yes! null로 취급해도 될 것 같음. 왜냐하면 null 이 아예 없음
# district_544M
data['district_544M'].value_counts()
data['district_544M'].isna().sum()


# cancelreason_3545846M
data['cancelreason_3545846M'].value_counts()
# rejectreason_755M
data['rejectreason_755M'].value_counts()
# rejectreasonclient_4145042M
data['rejectreasonclient_4145042M'].value_counts()
# cancelreason_3545846M 과 rejectreason_755M 는 P94_109_143라는 코드를 공유함.

# education_1138M
# postype_4733339M
# profession_152M


# 각자의 코드를 사용하는 범주형 변수
# credacc_status_367L
data['credacc_status_367L'].value_counts()
# inittransactioncode_279L
data['inittransactioncode_279L'].value_counts()
# status_219L
data['status_219L'].value_counts()
# credtype_587L
data['credtype_587L'].value_counts()
# familystate_726L
data['familystate_726L'].value_counts()


# 예상
# NUM_GROUP1 이전 신청서의 신청서 제출 SEQ수

# feature 제안
# num_group==0 인 신청서의 정보
# num_group==1 인 신청서의 정보
# 신청일자 기준 최근 n개월의 정보 agg
# num_group 기준 최근 n건의 정보 agg
#     범주형 변수 구분자별 agg (단 빈도가 너무 적은 것은 제외)
# case_id와 num_group1을 key로 해서 label을 붙여서 평가해보면 어떨까?
# 시계열적 정보를 반영하기 위해 nn을 사용해보면 어떨까?


################################################################################
FILE_NAME = "train_applprev"
DEPTH = 2
data = read_data(FILE_NAME, DEPTH)
appl_2 = data.copy()
describe_data(data)

# 0 case_id: case_id (postfix type : -) 2
# 1 cacccardblochreas_147M: 카드 블로킹 이유. (postfix type : Masking Categories) a55475b1
# 2 conts_type_509L: 이전 신청서의 개인 연락 유형. (postfix type : Unspecified Transform) PRIMARY_MOBILE
# 3 credacc_cards_status_52L: 이전 신청서의 신용 카드 상태. (postfix type : Unspecified Transform) CANCELLED
# 4 num_group1: num_group1 (postfix type : -) 0
# 5 num_group2: num_group2 (postfix type : -) 0

data["num_group1"].value_counts()
# appl_prev 와 같은 num_group을 공유하는지?
appl_1["case_id"].nunique()
# 1221522
appl_2["case_id"].nunique()
# 1221522

appl_1[["case_id", "num_group1"]].drop_duplicates().shape[0]
# 6525979
appl_2[["case_id", "num_group1"]].drop_duplicates().shape[0]
# 6525978

# 빠진걸 찾아보자
appl_1_unique = [
    tuple(row)
    for row in appl_1[["case_id", "num_group1"]].drop_duplicates().values.tolist()
]
appl_2_unique = [
    tuple(row)
    for row in appl_2[["case_id", "num_group1"]].drop_duplicates().values.tolist()
]
missing_value = set(appl_1_unique) - set(appl_2_unique)
print(missing_value)
appl_1[(appl_1['case_id'] == 1682638) & (appl_1['num_group1'] == 3)]
appl_2[(appl_2['case_id'] == 1682638) & (appl_2['num_group1'] == 3)]
# 하나면 case_id 째로 빼도 상관 없을 듯

temp_joined = appl_1.merge(appl_2, on=["case_id", "num_group1"], how="inner")
appl_2.shape[0]
# 14075487
temp_joined.shape[0]
# 14075487
# num_group1은 공유하는 것으로 보임

# 예상
# NUM_GROUP1 이전 신청서의 신청서 제출 SEQ
# NUM_GROUP2 이전 신청서의 ... 연락처 SEQ인지, 카드 SEQ인지..

data.groupby("case_id")["num_group2"].count().describe().astype(int)
data["num_group2"].value_counts()
# 1~3개의 정보가 있는 것으로 보임

data['cacccardblochreas_147M'].value_counts()
data["credacc_cards_status_52L"].value_counts(dropna=False)
data["conts_type_509L"].value_counts()
data[
    ["cacccardblochreas_147M", "credacc_cards_status_52L"]
].isna().value_counts().sort_index()
data[
    ["cacccardblochreas_147M", "credacc_cards_status_52L", "conts_type_509L"]
].isna().value_counts().sort_index()
data[~(data["cacccardblochreas_147M"].isna())&~(data["credacc_cards_status_52L"].isna())]
data[
    [
        'case_id',
        'num_group1',
        'cacccardblochreas_147M',
        'conts_type_509L',
        'credacc_cards_status_52L',
    ]
].drop_duplicates().shape[0]

# feature 제안
# num_group1에 정보를 붙여...
# 해석하기 어려워서, 컬럼으로 뽑아. 순서정보도 없는것같고...

################################################################################
################################################################################
# 개인정보
FILE_NAME = "train_person"
DEPTH = 1
data = read_data(FILE_NAME, DEPTH)
person_1 = data.copy()
describe_data(data)

data["num_group1"].value_counts()
# 0    1526659
# 1     757320
# 2     484214
# 3     181768
# 4      22453
# 5       1466
# 6         99
# 7          8
# 8          2
# 9          2
data[data["num_group1"] == 1]
data[data["num_group1"] == 0]

# 컬럼이 참 많다
# None이 참 많다.
data[data["num_group1"] == 0].isna().sum() / data[data["num_group1"] == 0].shape[0]

# num_group1!=0인 경우 None은 더 많다
data[data["num_group1"] != 0].isna().sum() / data[data["num_group1"] != 0].shape[0]
# 의미있는 컬럼은 아래정도...     na 비율
# personindex_1023L         0.443770 *
# persontype_1072L          0.004226
# persontype_792L           0.443770 *
# relationshiptoclient_415T 0.443770 *
# relationshiptoclient_642T 0.443153
# remitter_829L             0.443770 *
data[data["num_group1"] == 0].personindex_1023L.value_counts(dropna=False)
data[data["num_group1"] != 0].personindex_1023L.value_counts(dropna=False)
# data[data["num_group1"] == 0].persontype_1072L.value_counts(dropna=False)
# data[data["num_group1"] != 0].persontype_1072L.value_counts(dropna=False)
data[data["num_group1"] == 0].persontype_792L.value_counts(dropna=False)
data[data["num_group1"] != 0].persontype_792L.value_counts(dropna=False)
data[data["num_group1"] == 0].relationshiptoclient_415T.value_counts(dropna=False)
data[data["num_group1"] != 0].relationshiptoclient_415T.value_counts(dropna=False)
# data[data["num_group1"] == 0].relationshiptoclient_642T.value_counts(dropna=False)
# data[data["num_group1"] != 0].relationshiptoclient_642T.value_counts(dropna=False)
data[data["num_group1"] == 0].remitter_829L.value_counts(dropna=False)
data[data["num_group1"] != 0].remitter_829L.value_counts(dropna=False)
data[data["num_group1"] == 1].remitter_829L.value_counts(dropna=False)
data[data["num_group1"] >= 2].remitter_829L.value_counts(dropna=False)
describe('remitter_829L')
# korean(remit) == korean(transfer) 한국어로는 송금으로 동일
# remiter는 송금인이라는 뜻
na_ratio = (
    data[data["num_group1"] != 0].isna().sum() / data[data["num_group1"] != 0].shape[0]
)
non_na_cols = na_ratio[na_ratio==0].index
data[data["num_group1"] != 0][non_na_cols]


# NUM_GROUP1 신청인 SEQ

# feature 제안
# num_group1==0인 경우의 정보
# num_group1==1인 경우, 새로운 컬럼을 생성해서 붙이기
# 그 외 agg...


################################################################################
FILE_NAME = "train_person"
DEPTH = 2
data = read_data(FILE_NAME, DEPTH)
person_2 = data.copy()
describe_data(data)

data["num_group1"].value_counts()
# 0    1463928
# 1     175805
# 2       3529
# 3        145
# 4          3
# 대부분 num_group1==0인 경우가 많다
data[data["num_group1"] == 0].num_group2.value_counts()
# 본인인 경우(["num_group1"] == 0), num_group2==0 인 것의 비중 98%정도
# 2%만 num_group2가 여러개 달려있음
data[data["num_group1"] != 0].num_group2.value_counts()
# 본인이 아닌 경우, num_group2가 여러개 달려있는 경우가 더 많아짐 (10% 이상)

# appl_prev 와 같은 num_group을 공유하는지?
person_1["case_id"].nunique()
# 1526659
person_2["case_id"].nunique()
# 1435105

person_1[["case_id", "num_group1"]].drop_duplicates().shape[0]
# 2973991
person_2[["case_id", "num_group1"]].drop_duplicates().shape[0]
# 1561280

temp_joined = person_1.merge(person_2, on=["case_id", "num_group1"], how="inner")
person_2.shape[0]
# 1643410
temp_joined.shape[0]
# 1643410
# num_group1은 공유하는 것으로 보임
# person_2에는 person_1의 모든 case가 있는 것음 아님.
# 그러나 person_1과 person_2가 공유하는 case_id에 대해서는 동일한 갯수의 num_group1을 가지고 있음


data[data["num_group1"] == 0].addres_district_368M.value_counts()
data[data["num_group1"] != 0].addres_district_368M.value_counts()

data[data["num_group1"] == 0].addres_role_871L.value_counts(dropna=False)
data[data["num_group1"] != 0].addres_role_871L.value_counts(dropna=False)

data[data["num_group1"] == 0].conts_role_79M.value_counts(dropna=False)
data[data["num_group1"] != 0].conts_role_79M.value_counts(dropna=False)


data[data["num_group1"] == 0].relatedpersons_role_762T.value_counts(dropna=False)
data[data["num_group1"] != 0].relatedpersons_role_762T.value_counts(dropna=False)


# 1 addres_district_368M: 개인 주소의 지역. (postfix type : Masking Categories) a55475b1 **
# 2 addres_role_871L: 개인 주소의 역할. (postfix type : Unspecified Transform) CONTACT **
# 3 addres_zip_823M: 주소의 우편번호. (postfix type : Masking Categories) a55475b1 **
# 4 conts_role_79M: 개인의 연락 역할 유형. (postfix type : Masking Categories) a55475b1
# 5 empls_economicalst_849M: 개인의 경제적 상태 (num_group1 - 개인, num_group2 - 고용). (postfix type : Masking Categories) a55475b1
# 6 empls_employedfrom_796D: 고용 시작 (num_group1 - 개인, num_group2 - 고용). (postfix type : Transform Date) 2018-06-15
# 7 empls_employer_name_740M: 고용주의 이름 (num_group1 - 개인, num_group2 - 고용). (postfix type : Masking Categories) a55475b1
# 8 num_group1: num_group1 (postfix type : -) 0
# 9 num_group2: num_group2 (postfix type : -) 0
# 10 relatedpersons_role_762T: 고객의 관련된 사람의 관계 유형(num_group1 - 사람, num_group2 - 관련된 사람). (postfix type : Unspecified Transform)


data
data[data["case_id"] == 6]
data[data["case_id"] == 22]
# num_group1, num_group2
# 0, 0 본인거주지
# 0, 1 본인회사
# 1, 0 관련인거주지
# 1, 1 관련인회사
data[data["case_id"] == 2702502]


narole = data[
    (data["num_group1"] == 0)
    & (data["num_group2"] != 0)
    & (data["addres_role_871L"].isna())
]
for col in narole.columns:
    print(col, narole[col].nunique())
data[data["case_id"] == 578] # 회사와 관련된 seq 증가?


narole = data[data['addres_role_871L'].isna()]
for col in narole.columns:
    print(col, narole[col].nunique())

narole = data[data['addres_role_871L'].isna()&(data['conts_role_79M']=='a55475b1')]
for col in narole.columns:
    print(col, narole[col].nunique())

narole = data[data['addres_role_871L'].isna() & (data['conts_role_79M'] != 'a55475b1')]
for col in narole.columns:
    print(col, narole[col].nunique())

narole = data[~data['addres_role_871L'].isna() & (data['conts_role_79M'] == 'a55475b1')]
for col in narole.columns:
    print(col, narole[col].nunique())

narole = data[~data['addres_role_871L'].isna() & (data['conts_role_79M'] != 'a55475b1')]
for col in narole.columns:
    print(col, narole[col].nunique())


# NUM_GROUP1 신청인 SEQ
# NUM_GROUP2 ?
# 하나의 table 안에서 num_group2 의미가 달라질 수 있음
# 예를들어 addres_role_871L 가 있는 경우/없는 경우
# (num_group1 - 개인, num_group2 - 고용) 설명이 붙어있는경우와 그렇지 않은 경우, 컬럼에서 쓰는 num_group2의 의미가 다를 수 있음
# 각각 seq를 매기고 cartesian product를 만든 것 같음. 진짜 개짱남.
# 컬럼은 서로 연관되어 있을수도 있고 아닐수도 있음... 이게 무슨일이야

#  In case num_group1 or num_group2 stands for person index (this is clear with predictor definitions)
#  the zero index has special meaning.
#  When num_groupN=0 it is the applicant (the person who applied for a loan)


################################################################################
################################################################################
# 직불카드정보
FILE_NAME = "train_debitcard"
DEPTH = 1
data = read_data("train_debitcard", 1)
describe_data(data)

# NUM_GROUP1 카드 SEQ?

################################################################################
################################################################################
# 예금정보
FILE_NAME = "train_deposit"
DEPTH = 1
data = read_data("train_deposit", 1)
describe_data(data)

# NUM_GROUP1 신청인 SEQ? 계좌 SEQ?

################################################################################
################################################################################
# 기타정보
FILE_NAME = "train_other"
DEPTH = 1
data = read_data("train_other", 1)
describe_data(data)

# NUM_GROUP1 모두 0 의미없음


################################################################################
################################################################################
# CB정보 A
data = read_data("train_credit_bureau_a", 1)
describe_data(data)
data['case_id'].nunique()
data['num_group1'].value_counts()
# num_group1 의 의미 : 보유 계좌정보 SEQ, 이 계좌는 현재 유효할 수도, 그렇지 않을수도 있음
# 활성 / 종료된 으로 컬럼이 나누어져 있는데, 하나의 컬럼으로 합치고 valid_yn 같은 컬럼으로 구분하는게 좋을듯


# 0 case_id: case_id (postfix type : -) 29427
# 1 collater_typofvalofguarant_298M: 활성 계약의 담보 평가 유형. (postfix type : Masking Categories) 9a0c095e
# 2 collater_typofvalofguarant_407M: 종료된 계약의 담보 평가 유형. (postfix type : Masking Categories) a55475b1
# 3 collater_valueofguarantee_1124L: 활성 계약의 담보 가치. (postfix type : Unspecified Transform) 0.0
# 4 collater_valueofguarantee_876L: 종료된 계약의 담보 가치. (postfix type : Unspecified Transform) 0.0
# 5 collaterals_typeofguarante_359M: 종료된 계약을 위한 담보로 사용된 담보 유형. (postfix type : Masking Categories) a55475b1
# 6 collaterals_typeofguarante_669M: 활성 계약의 담보 유형. (postfix type : Masking Categories) c7a5ad39
# 7 num_group1: num_group1 (postfix type : -) 0
# 8 num_group2: num_group2 (postfix type : -) 0
# 9 pmts_dpd_1073P: 활성 계약의 연체 지불 일 수(num_group1 - 기존 계약, num_group2 - 지불). (postfix type : Transform DPD (Days Past Due)) 0.0
# 10 pmts_dpd_303P: 신용 레포트에 따른 종료된 계약의 연체 지불 일 수(num_group1 - 종료된 계약, num_group2 - 지불). (postfix type : Transform DPD (Days Past Due)) 0.0
# 11 pmts_month_158T: 폐쇄된 계약의 지불 월(num_group1 - 기존 계약, num_group2 - 지불). (postfix type : Unspecified Transform) 2.0
# 12 pmts_month_706T: 활성 계약의 지불 월(num_group1 - 종료된 계약, num_group2 - 지불). (postfix type : Unspecified Transform) 2.0
# 13 pmts_overdue_1140A: 활성 계약의 연체 지불(num_group1 - 기존 계약, num_group2 - 지불). (postfix type : Transform Amount) 0.0
# 14 pmts_overdue_1152A: 폐쇄된 계약의 연체 지불(num_group1 - 종료된 계약, num_group2 - 지불). (postfix type : Transform Amount) 0.0
# 15 pmts_year_1139T: 활성 계약의 지불 연도(num_group1 - 기존 계약, num_group2 - 지불). (postfix type : Unspecified Transform) 2019.0
# 16 pmts_year_507T: 폐쇄된 신용 계약의 지불 연도(num_group1 - 종료된 계약, num_group2 - 지불). (postfix type : Unspecified Transform) 2007.0
# 17 subjectroles_name_541M: 종료된 신용 계약의 주제 역할 이름 (num_group1 - 해지된 계약, num_group2 - 주제 역할). (postfix type : Masking Categories) a55475b1
# 18 subjectroles_name_838M: 활성 신용 계약의 주제 역할 이름 (num_group1 - 기존 계약, num_group2 - 주제 역할).

data = read_data("train_credit_bureau_a", 2, num_files=2)
data = read_data("train_credit_bureau_a", 2)
describe_data(data)
# num_group1 의 의미 : 보유 계좌정보 SEQ, 이 계좌는 현재 유효할 수도, 그렇지 않을수도 있음
# num_group2 의 의미
    # -- 보유 계좌의 담보 유형의 종류
    # -- 보유 계좌의 지불 seq
    # -- 보유 계좌의 subjectrole... 이 뭘까?
# 활성 / 종료된 으로 컬럼이 나누어져 있는데, 하나의 컬럼으로 합치고 valid_yn 같은 컬럼으로 구분하는게 좋을듯
data[data["case_id"] == 29427]
data[data["case_id"] == 2688744]


################################################################################
# CB정보 B
data = read_data("train_credit_bureau_b", 1)
describe_data(data)

# CB정보 B
data = read_data("train_credit_bureau_b", 2)
describe_data(data)

# 이 나라에는 CB사가 2개가 있고,
# CB사 A는 연체, 지불, 담보 등 다양한 정보를 가지고 있고
# CB사 B는 연체 정보만 보유함
# 정보 가공 방식은 거의 유사함