import polars as pl
from dataset.datainfo import RawInfo
from dataset.feature.feature import *
from typing import Dict, Iterable

pl.Config.set_streaming_chunk_size(1000)

infos = RawInfo()
base_df = infos.read_raw("base")
applprev_df = infos.read_raw("applprev", depth=1)
base_df.date_decision.value_counts().sort_index()

# applprev_df.creationdate_885D.str[:4].value_counts()

raw_cols: Dict[str, Column] = {
    col: Column(name=col, data_type=str(type), depth=1)
    for col, type in applprev_df.dtypes.items()
    if col not in ('case_id', 'MONTH')
}


## filter
filter_cols: Dict[str, Column] = {
    name: col
    for name, col in raw_cols.items()
    if (col.postfix in ('L', 'T', 'M') and col.data_type == 'object')
}

filters: List[Filter] = [
    Filter(columns=[col], logic=f"{col} = '{val}'", value=val)
    for col in filter_cols.values()
    for val in applprev_df[col.name].value_counts().index[:10]
    if val != 'a55475b1'
]

for col in filter_cols.values():
    hasnull = any(applprev_df[col.name].isnull())
    if col.data_type == 'object':
        hasa55475b1 = any(applprev_df[col.name] == 'a55475b1')
        if hasnull and hasa55475b1:
            filters.append(
                Filter(
                    columns=[col],
                    logic=f"{col} is null or {col}  = 'a55475b1'",
                    value='a55475b1',
                )
            )
            filters.append(
                Filter(
                    columns=[col],
                    logic=f"{col} is not null and {col}  != 'a55475b1'",
                    value='a55475b1',
                )
            )
        elif hasnull:
            filters.append(Filter(columns=[col], logic=f"{col} is null"))
            filters.append(Filter(columns=[col], logic=f"{col} is not null"))
        elif hasa55475b1:
            filters.append(
                Filter(columns=[col], logic=f"{col} = 'a55475b1'", value='a55475b1')
            )
            filters.append(
                Filter(columns=[col], logic=f"{col} != 'a55475b1'", value='a55475b1')
            )
        elif hasgezero:
            filters.append(Filter(columns=[col], logic=f"{col} > 0"))
            filters.append(Filter(columns=[col], logic=f"{col} <= 0"))
    else:
        hasgezero = any(applprev_df[col.name] <= 0)
        if hasnull and hasgezero:
            filters.append(Filter(columns=[col], logic=f"{col} is null"))
            filters.append(Filter(columns=[col], logic=f"{col} is not null"))
            filters.append(Filter(columns=[col], logic=f"{col} > 0", value=0))
            filters.append(Filter(columns=[col], logic=f"{col} <= 0", value=0))
        if hasnull:
            filters.append(Filter(columns=[col], logic=f"{col} is null"))
            filters.append(Filter(columns=[col], logic=f"{col} is not null"))
        if hasgezero:
            filters.append(Filter(columns=[col], logic=f"{col} > 0", value=0))
            filters.append(Filter(columns=[col], logic=f"{col} <= 0", value=0))

# list of fivonacci numbers
def fibonacci(target_series: pd.Series,):
    fibonacci = [1, 2]
    while fibonacci[-1] < max(target_series):
        fibonacci.append(fibonacci[-1] + fibonacci[-2])
    fibonacci = fibonacci[1:]
    return fibonacci


filters += [
    Filter(columns=[raw_cols['num_group1']], logic=f"num_group1 < {val}", value=list(range(val)))
    for val in fibonacci(applprev_df['num_group1'])
]
filters += [
    Filter(columns=[raw_cols['num_group1']], logic=f"num_group1 = {val}", value=val)
    for val in list(range(0, 3))
]

period_cols = {
    name: col
    for name, col in raw_cols.items()
    if col.postfix == 'D' and name in ('creationdate_885D')
}


def date_diff(x: pd.Series, date='2020-10-19'):
    if x.dtype == 'object':
        x = pd.to_datetime(x)
    return (pd.to_datetime(date) - x).dt.days

period_filters: List[Filter] = [
    Filter(
        columns=[col],
        logic=f"date(date_decision)-date({col}) < {val}",
        value=list(range(val)),
    )
    for col in period_cols.values()
    for val in fibonacci(date_diff(applprev_df['creationdate_885D']))
]
# [col for col in period_cols.values()]
filters += period_filters

len(filters)

## agg
# 여부, 년도, 각 날짜별 datediff

# aggs = [
#     Agg(columns=[col], logic="count({0})")
#     for col in raw_cols.values()
# ]

aggs = [
    GroupAgg(columns=[col], logic="sum({0})", data_type='int64', aggmart_logics=["count({0})"])
    for col in raw_cols.values()
]


# numeric_aggregater = [
#     'sum({0})',
#     'avg({0})',
#     'min({0})',
#     'max({0})',
#     'stddev({0})',
# ]

# for agg in numeric_aggregater:
#     aggs += [
#         Agg(columns=[col], logic=agg)
#         for col in raw_cols.values()
#         if col.data_type in ('int64', 'float64')
#     ]

# categorycal_aggregater = [
#     'count(distinct {0})',
#     'max({0})',
# ]

# for agg in categorycal_aggregater:
#     aggs += [
#         Agg(columns=[col], logic=agg)
#         for col in raw_cols.values()
#         if col.data_type in ('object')
#         and col.postfix in ('L', 'T', 'M')
#     ]

# date_aggregater = [
#     'max(DATE(date_decision) - DATE({0}))',
#     'min(DATE(date_decision) - DATE({0}))',
#     'avg(DATE(date_decision) - DATE({0}))',
#     'stddev(DATE(date_decision) - DATE({0}))',
# ]

# for agg in date_aggregater:
#     aggs += [
#         Agg(columns=[col], logic=agg)
#         for col in raw_cols.values()
#         if col.data_type in ('object')
#         and col.postfix in ('D')
#     ]

len(aggs)

# TODO : no filter case


features = [
    Feature(
        depth=1,
        data_type="int64",
        topic="applprev_1",
        agg=agg,
        filters=[filter],
    )
    for _, agg in enumerate(aggs)
    for _, filter in enumerate(filters)
    if agg.columns[0] != filter.columns[0]
]

# save features as yaml file

[feature.to_dict() for feature in features]
[feature.__dict__ for feature in features]


import json
with open('data/applprev_1_0.json','w') as f:
    json.dump([feature.to_dict() for feature in features], f)

# load features from json file
with open('data/applprev_1_0.json', 'r') as f:
    features = [Feature.from_dict(feature) for feature in json.load(f)]

len(features)

from collections import defaultdict

class FeatureBuilder:
    def __init__(self, features: List[Feature]):
        self.features = features

    @property
    def grouped_features(self) -> List[Feature]:
        return [
            feat for feat in self.features if isinstance(feat.agg, GroupAgg)
        ]

    @property
    def non_grouped_features(self) -> List[Feature]:
        return [
            feat for feat in self.features if isinstance(feat.agg, Agg)
        ]

    def select_query(self) -> str:
        return '\n, '.join(
            [f'{feat.query} as {feat.name}' for feat in self.non_grouped_features]
        )

    def select_grouped_query(self) -> str:
        return '\n, '.join(
            [f'{feat.query} as {feat.name}' for feat in self.grouped_features]
        )

    def with_select_query(self) -> str:
        aggmart_columns = [feat.agg.aggmart_columns for feat in self.grouped_features]
        flattened_list = [item for columns in aggmart_columns for item in columns]
        aggmart_columns = set(flattened_list)

        return '\n, '.join(
            [f'{column.query} as {column.name}' for column in aggmart_columns]
        )

    def group_filters(self) -> List[Filter]:
        flattened_filter = [
            filter for feat in self.grouped_features for filter in feat.filters
        ]
        filter_value_pairs = [
            (filter.columns[0].name, filter.value) for filter in flattened_filter
        ]
        result = defaultdict(set)
        for key, value in filter_value_pairs:
            if isinstance(value, Iterable):
                value.remove( None )
                result[key].update(value)
            elif value is not None:
                result[key].add(value)

        return result

    def group_filters_query(self) -> str:
        grouped_filters = self.group_filters()

        return '\n, '.join(
            [
                f'case when {key} in ({", ".join(value_set)} then {key} else "elsevalue" end as {key}'
                for key, value_set in grouped_filters.items()
            ]
        )
    
# def _required_columns(self) -> List[Column]:
#     required_columns = []
#     for feature in self.features:
#         required_columns.extend(self._required_columns_for_feature(feature))
#     required_columns = set(required_columns)
#     return required_columns

# def _required_columns_for_feature(self, feature) -> List[Column]:
#     if not isinstance(feature.agg, GroupAgg):
#         return list()


fb = FeatureBuilder(features[:1000])
print(fb.group_filters())

features[0].query
[feature.query for feature in features]

data = pl.read_parquet(
    'data/home-credit-credit-risk-model-stability/parquet_files/train/train_applprev_1_0.parquet',
    use_pyarrow=True
)
base = pl.read_parquet(
    'data/home-credit-credit-risk-model-stability/parquet_files/train/train_base.parquet',
    use_pyarrow=True,
)
frame = data.join(base.select(['case_id', 'date_decision']), on='case_id', how='inner')


# all_query[57:207]
def get_query(frame, all_query):
    return
    res = pl.SQLContext(frame=frame).execute(
        f"""
        SELECT frame.case_id
         , {all_query}
        from frame
        group by frame.case_id
        order by frame.case_id
        """,
        eager=True
    )
    return res


all_query = '\n, '.join([feature.query for feature in features[:1000]])
all_agg_query = '\n, '.join(
    [feature.agg.aggmart_columns for feature in features[:1000]]
)
print(all_query)
out_res = get_query(frame, all_query)
out_res.head(10)

# data.select('case_id').n_unique()
len(base_df)
# tax_registry_a_df = infos.read_raw("tax_registry_a", depth=1)
# tax_registry_b_df = infos.read_raw("tax_registry_b", depth=1)
# tax_registry_c_df = infos.read_raw("tax_registry_c", depth=1)
# out = pd.concat(
#     [
#         tax_registry_a_df['case_id'].drop_duplicates(),
#         tax_registry_b_df['case_id'].drop_duplicates(),
#         tax_registry_c_df['case_id'].drop_duplicates(),
#     ]
# )
# out.drop_duplicates()
