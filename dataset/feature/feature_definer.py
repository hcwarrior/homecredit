import pandas as pd
from dataset.datainfo import RawInfo, RawReader
from dataset.feature.feature import *
from dataset.const import TOPICS
from typing import Dict, List
import json


class FeatureDefiner:
    RAW_INFO = RawInfo()

    def __init__(self, topic: str, numgroup: str = 'num_group1' ,period_cols: List[str] = None):
        self.topic: str = topic
        self.rawdata: pd.DataFrame = self.RAW_INFO.read_raw(self.topic, depth=1)
        self.raw_cols: Dict[str, Column] = {
            col: Column(name=col, data_type=str(type))
            for col, type in self.rawdata.dtypes.items()
            if col not in ('case_id')
        }
        self.numgroup: str = numgroup
        self.period_cols: List[str] = period_cols
        self._update_integer_data_type()

    def _update_integer_data_type(self):
        for col in self.raw_cols.values():
            if col.data_type == 'float64' and all(
                self.rawdata[col.name].apply(lambda x: x.is_integer())
            ):
                self.raw_cols[col.name].data_type = 'int64'

    def define_features(self):
        aggs = self.define_aggs()
        filters = self.define_filters(self.numgroup, self.period_cols)
        filters += [None]
        features: List[Feature] = [
            Feature(
                data_type=agg.data_type,
                topic=self.topic,
                agg=agg,
                filters=[filter],
            )
            for agg in aggs
            for filter in filters
            if filter is None or (agg.columns[0] != filter.columns[0])
        ]
        return features, aggs, filters

    def define_filters(self, numgroup: str = 'num_group1', period_cols: List[str] = None):
        filters = self.categorical_filters()
        if period_cols is not None:
            filters += self.period_filters(period_cols)
        filters += self.null_filters()
        if numgroup is not None:
            filters += self.numgroup_filters(numgroup)
        return filters

    def categorical_filters(self):
        filter_cols: Dict[str, Column] = {
            name: col
            for name, col in self.raw_cols.items()
            if (col.postfix in ('L', 'T', 'M') and col.data_type == 'object')
        }

        filters: List[Filter] = [
            Filter(columns=[col], logic=f"{col} = '{val}'")
            for col in filter_cols.values()
            for val in self.rawdata[col.name].value_counts().index[:10]
            if val != 'a55475b1'
        ]
        return filters

    def period_filters(self, period_cols: List[str]):
        period_cols: Dict[str, Column] = {
            name: col
            for name, col in self.raw_cols.items()
            if col.postfix == 'D' and name in (period_cols)
        }

        filters: List[Filter] = [
            Filter(
                columns=[col],
                logic=f"date(date_decision)-date({col}) < {val}",
            )
            for col in period_cols.values()
            for val in self.fibonacci(self.date_diff(self.rawdata[col.name]))[3:]
        ]
        return filters

    def null_filters(self):
        filters: List[Filter] = []
        for col in self.raw_cols.values():
            hasnull = any(self.rawdata[col.name].isnull())
            if col.data_type == 'object':
                hasa55475b1 = any(self.rawdata[col.name] == 'a55475b1')
                if hasnull and hasa55475b1:
                    filters.append(Filter([col],f"{col} is null or {col} = 'a55475b1'"))
                    filters.append(Filter([col],f"{col} is not null and {col} != 'a55475b1'"))
                elif hasnull:
                    filters.append(Filter([col], f"{col} is null"))
                    filters.append(Filter([col], f"{col} is not null"))
                elif hasa55475b1:
                    filters.append(Filter([col], f"{col} = 'a55475b1'"))
                    filters.append(Filter([col], f"{col} != 'a55475b1'"))
            else:
                hasgezero = any(self.rawdata[col.name] <= 0)
                if hasnull and hasgezero:
                    filters.append(Filter([col], f"{col} is null"))
                    filters.append(Filter([col], f"{col} is not null"))
                    filters.append(Filter([col], f"{col} > 0"))
                    filters.append(Filter([col], f"{col} <= 0"))
                elif hasnull:
                    filters.append(Filter([col], f"{col} is null"))
                    filters.append(Filter([col], f"{col} is not null"))
                elif hasgezero:
                    filters.append(Filter([col], f"{col} > 0"))
                    filters.append(Filter([col], f"{col} <= 0"))
        return filters

    def numgroup_filters(self, numgroup: str):
        filters: List[Filter] = []
        filters += [
            Filter(columns=[self.raw_cols[numgroup]], logic=f"{numgroup} < {val}")
            for val in self.fibonacci(self.rawdata[numgroup])
        ]
        filters += [
            Filter(columns=[self.raw_cols[numgroup]], logic=f"{numgroup} = {val}")
            for val in list(range(0, 2))
        ]
        return filters

    def define_aggs(self):
        aggs: List[Agg] = [
            Agg(columns=[col], logic="count({0})", data_type='int')
            for col in self.raw_cols.values()
        ]
        aggs += self.numeric_aggs()
        aggs += self.categorical_aggs()
        aggs += self.date_aggs()
        return aggs

    def numeric_aggs(self):
        aggs: List[Agg] = []
        numeric_cols: Dict[str, Column] = {
            name: col
            for name, col in self.raw_cols.items()
            if col.data_type in ('int64', 'float64')
        }
        simple_numeric_aggregater = [
            'sum({0})',
            'min({0})',
            'max({0})',
        ]
        aggs += [
            Agg(columns=[col], logic=agg, data_type=col.data_type.replace('64', ''))
            for col in numeric_cols.values()
            for agg in simple_numeric_aggregater
        ]
        complex_numeric_aggregater = [
            'avg({0})',
            'stddev({0})',
        ]
        aggs += [
            Agg(columns=[col], logic=agg, data_type='float')
            for col in self.raw_cols.values()
            for agg in complex_numeric_aggregater
        ]
        return aggs

    def categorical_aggs(self):
        aggs: List[Agg] = []
        categorical_cols: Dict[str, Column] = {
            name: col
            for name, col in self.raw_cols.items()
            if col.data_type in ('object')
        }
        aggs += [
            Agg(columns=[col], logic='count(distinct {0})', data_type='int')
            for col in categorical_cols.values()
        ]
        aggs += [
            Agg(columns=[col], logic='max({0})', data_type='string')
            for col in categorical_cols.values()
        ]
        return aggs

    def date_aggs(self):
        aggs: List[Agg] = []
        date_cols: Dict[str, Column] = {
            name: col
            for name, col in self.raw_cols.items()
            if col.data_type in ('object') and col.postfix in ('D')
        }
        date_aggregater = [
            'max(DATE(date_decision) - DATE({0}))',
            'min(DATE(date_decision) - DATE({0}))',
            'avg(DATE(date_decision) - DATE({0}))',
            'stddev(DATE(date_decision) - DATE({0}))',
        ]
        aggs += [
            Agg(columns=[col], logic=agg, data_type='int')
            for col in date_cols.values()
            for agg in date_aggregater
        ]
        return aggs

    @staticmethod
    def fibonacci(target_series: pd.Series):
        if target_series.dtype == 'object':
            raise ValueError("target_series should be numeric")
        max_value = max(target_series)

        fibonacci = [1, 2]
        while fibonacci[-1] < max_value:
            fibonacci.append(fibonacci[-1] + fibonacci[-2])
        fibonacci = fibonacci[1:]
        return fibonacci

    @staticmethod
    def date_diff(x: pd.Series, date='2020-10-19'):
        if x.dtype == 'object':
            x = pd.to_datetime(x)
        return (pd.to_datetime(date) - x).dt.days

    @staticmethod
    def save_json(features: List[Feature], filename: str):
        with open(filename, 'w') as f:
            json.dump({feature.name: feature.to_dict() for feature in features}, f)
