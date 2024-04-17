import pandas as pd
import os
from pathlib import Path
from glob import glob
from dataclasses import dataclass
from dataset.const import TOPICS
from typing import List, Union


class Column:

    def __init__(
        self, depth: int, data_type: str, query: str = None, name: str = None
    ) -> None:
        self.depth = depth
        self.data_type = data_type
        self.query = query
        self.name = name if name else self._set_name_using_query(query)
        self.validate_data_type()

    def validate_data_type(self):
        if self.data_type not in ["int64", "float64", "object"]:
            raise ValueError(
                f"Invalid data_type: {self.data_type}. Available data_types: "
            )

    def __str__(self) -> str:
        return self.name

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Column):
            return False
        return (
            self.depth == value.depth
            and self.data_type == value.data_type
            and self.query == value.query
            and self.name == value.name
        )

    def __hash__(self) -> int:
        return hash((self.depth, self.data_type, self.query, self.name))

    @property
    def postfix(self):
        return self.name[-1]

    @staticmethod
    def _set_name_using_query(formated_query: str):
        if formated_query is None:
            return None

        formated_query = formated_query.replace("<", "lt")
        formated_query = formated_query.replace("<=", "le")
        formated_query = formated_query.replace(">", "gt")
        formated_query = formated_query.replace(">=", "ge")
        formated_query = formated_query.replace("=", "eq")
        formated_query = formated_query.replace("/", "div")
        formated_query = formated_query.replace("*", "mul")
        formated_query = formated_query.replace("case when ", "_if_")
        formated_query = formated_query.replace("else null end", "")
        formated_query = formated_query.replace("(", "_")
        formated_query = formated_query.replace(")", "_")
        formated_query = formated_query.replace("'", "")
        formated_query = formated_query.replace(" and ", "")
        formated_query = formated_query.replace(" else ", "_")
        formated_query = formated_query.replace(" ", "_")
        return formated_query

    def to_dict(self):
        return {
            "depth": self.depth,
            "data_type": self.data_type,
            "query": self.query,
            "name": self.name,
        }


@dataclass
class Topic:
    name: str
    max_depth: int
    columns: List[Column]

    def __post_init__(self):
        self.validate_topic()

    def validate_topic(self):
        if self.name not in TOPICS:
            raise ValueError(f"Invalid topic: {self.name}. Available topics: {TOPICS}")


@dataclass
class Element:
    columns: List[Column]
    logic: str

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Element):
            return False
        return self.columns == value.columns and self.logic == value.logic
    
    def __str__(self) -> str:
        return self.logic.format(*map(str, self.columns))

    @property
    def query(self):
        return self.logic.format(*map(str, self.columns))


class Filter(Element):
    def __init__(self, columns: List[Column], logic: str, value: Union[str, int, List[int], List[str]] = None):
        super().__init__(columns, logic)
        self.value = value

    def __str__(self) -> str:
        return '(' + self.logic.format(*map(str, self.columns)) + ')'

    def to_dict(self):
        return {
            "columns": [column.to_dict() for column in self.columns],
            "logic": self.logic,
            "value": self.value,
        }

    @staticmethod
    def from_dict(data: dict):
        columns = [Column(**column) for column in data["columns"]]
        data.pop("columns")
        return Filter(columns=columns, **data)


class Agg(Element):
    def __init__(self, columns: List[Column], logic: str, data_type: str):
        super().__init__(columns, logic)
        self.data_type = data_type

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Agg):
            return False
        return super().__eq__(value) and self.data_type == value.data_type

    def to_dict(self):
        return {
            "columns": [column.to_dict() for column in self.columns],
            "logic": self.logic,
            "data_type": self.data_type,
        }

    @staticmethod
    def from_dict(data: dict):
        columns = [Column(**column) for column in data["columns"]]
        data.pop("columns")
        return Agg(columns=columns, **data)


class Feature(Column):
    def __init__(
            self, depth: int, data_type: str, topic: str, agg: Agg, filters: List[Filter]
        ):
        super().__init__(depth, data_type)
        self.topic = topic
        self.agg = agg
        self.filters = filters
        self.query = self._init_query()
        self.name = self._set_name_using_query(self.query)

    def _init_query(self):
        if not isinstance(self.agg, GroupAgg):
            column_names = [column.name for column in self.agg.columns]
        else:
            column_names = [column.name for column in self.agg.aggmart_columns]

        filter_condition = " and ".join(map(str, self.filters)) if len(self.filters) > 0 else "1 = 1"
        filtered_columns = [
            f'case when {filter_condition} then {column} else null end'
            for column in column_names
        ]
        return self.agg.logic.format(*filtered_columns)

    def to_dict(self):
        return {
            "depth": self.depth,
            "data_type": self.data_type,
            "topic": self.topic,
            "agg": self.agg.to_dict(),
            "filters": [filter.to_dict() for filter in self.filters],
        }

    @staticmethod
    def from_dict(data: dict):
        if data["agg"].get("aggmart_logics"):
            agg = GroupAgg.from_dict(data["agg"])
        else:
            agg = Agg.from_dict(data["agg"])
        filters = [Filter.from_dict(filter) for filter in data["filters"]]
        data.pop("agg")
        data.pop("filters")
        return Feature(agg=agg, filters=filters, **data)


class GroupAgg(Agg):
    def __init__(
        self,
        columns: List[Column],
        logic: str,
        data_type: str,
        aggmart_logics: str = None,
    ):
        super().__init__(columns, logic, data_type)
        self.aggmart_logics = aggmart_logics

    @property
    def aggmart_columns(self) -> List[Column]:
        if not self.aggmart_logics:
            return list()
        queries = [
            logic.format(*map(str, self.columns)) for logic in self.aggmart_logics
        ]
        return [Column(-1, self.data_type, query=query) for query in queries]

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, GroupAgg):
            return False
        return super().__eq__(value) and self.aggmart_logics == value.aggmart_logics

    def to_dict(self):
        return {
            "columns": [column.to_dict() for column in self.columns],
            "logic": self.logic,
            "data_type": self.data_type,
            "aggmart_logics": self.aggmart_logics,
        }

    @staticmethod
    def from_dict(data: dict):
        columns = [Column(**column) for column in data["columns"]]
        data.pop("columns")
        return GroupAgg(columns=columns, **data)


## test code
sample_agg = Agg(
    columns=[Column(name="test1", data_type="int64", depth=1)],
    logic="sum({0})",
    data_type="int64",
)
sample_filter = Filter(
    columns=[Column(name="test2", data_type="int64", depth=1)],
    logic="{0} = 1",
)
sample_agg.query

sample_feature = Feature(
    depth=1,
    data_type="int64",
    topic="applprev_1",
    agg=sample_agg,
    filters=[sample_filter, sample_filter],
)
sample_feature.query
sample_feature.name

## test code
sample_agg = GroupAgg(
    columns=[Column(name="test1", data_type="int64", depth=1)],
    logic="sum({0})",
    aggmart_logics=["sum({0})"],
    data_type="int64",
)
sample_filter = Filter(
    columns=[Column(name="test2", data_type="int64", depth=1)],
    logic="{0} = 1",
)
sample_agg.aggmart_columns[0].name
sample_agg.aggmart_columns[0].query

sample_feature = Feature(
    depth=1,
    data_type="int64",
    topic="applprev_1",
    agg=sample_agg,
    filters=[sample_filter, sample_filter],
)
sample_feature.query
sample_feature.name

## test code
sample_agg = GroupAgg(
    columns=[Column(name="test1", data_type="int64", depth=1)],
    logic="sum({0})/sum({1})",
    aggmart_logics=["sum({0})", "count({0})"],
    data_type="int64",
)
sample_filter = Filter(
    columns=[Column(name="test2", data_type="int64", depth=1)],
    logic="{0} = 1",
)
sample_agg.aggmart_columns[1].name
sample_agg.aggmart_columns[1].query

sample_feature = Feature(
    depth=1,
    data_type="int64",
    topic="applprev_1",
    agg=sample_agg,
    filters=[sample_filter, sample_filter],
)
sample_feature.query
sample_feature.name
