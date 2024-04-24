from dataclasses import dataclass
from dataset.const import QUERY_FORMAT_REPLACEMENTS
from typing import List


class Column:
    """
    Column class to represent a column in the dataset.
    """

    def __init__(
        self, data_type: str, query: str = None, name: str = None
    ) -> None:
        self.data_type = data_type
        self.query = query
        self.name = name if name else self._set_name_using_query(query)

    def __str__(self) -> str:
        return self.name

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Column):
            return False
        return (self.data_type == value.data_type
            and self.query == value.query
            and self.name == value.name
        )

    def __hash__(self) -> int:
        return hash((self.data_type, self.query, self.name))

    @property
    def postfix(self):
        return self.name[-1]

    @staticmethod
    def _set_name_using_query(formated_query: str):
        if formated_query is None:
            return None

        for key, value in QUERY_FORMAT_REPLACEMENTS:
            formated_query = formated_query.replace(key, value)
        return formated_query.lower()

    def to_dict(self):
        return self.__dict__

@dataclass
class Element:
    """
    Element class to represent a filter or aggregation in the dataset.
    """
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
    def __init__(self, columns: List[Column], logic: str):
        super().__init__(columns, logic)

    def __str__(self) -> str:
        return '(' + self.logic.format(*map(str, self.columns)) + ')'

    def to_dict(self):
        return {
            "columns": [column.to_dict() for column in self.columns],
            "logic": self.logic,
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
    """
    Feature class to represent a feature in the dataset.

    Args:
        data_type (str): The data type of the feature.
        topic (str): The topic of the feature.
        agg (Agg): The aggregation object associated with the feature.
        filters (List[Filter]): The list of filters applied to the feature.

    Attributes:
        topic (str): The topic of the feature.
        agg (Agg): The aggregation object associated with the feature.
        filters (List[Filter]): The list of filters applied to the feature.
        query (str): The SQL query generated based on the aggregation and filters.
        name (str): The name of the feature derived from the query.

    Methods:
        _init_query: Initializes the SQL query based on the aggregation and filters.
        to_dict: Converts the Feature object to a dictionary.
        from_dict: Creates a Feature object from a dictionary.

    """

    def __init__(
            self, data_type: str, topic: str, agg: Agg, filters: List[Filter]
        ):
        super().__init__(data_type)
        self.topic = topic
        self.agg = agg
        self.filters = [filter for filter in filters if filter is not None]
        self.query = self._init_query()
        self.name = self._set_name_using_query(self.query)

    def _init_query(self):
        column_names = [column.name for column in self.agg.columns]

        filter_condition = " and ".join(map(str, self.filters)) if len(self.filters) > 0 else "1 = 1"
        filtered_columns = [
            f'case when {filter_condition} then {column} else null end'
            for column in column_names
        ]
        return self.agg.logic.format(*filtered_columns)

    def to_dict(self):
        return {
            "data_type": self.data_type,
            "topic": self.topic,
            "agg": self.agg.to_dict(),
            "filters": [filter.to_dict() for filter in self.filters],
        }

    @staticmethod
    def from_dict(data: dict):
        agg = Agg.from_dict(data["agg"])
        filters = [Filter.from_dict(filter) for filter in data["filters"]]
        data.pop("agg")
        data.pop("filters")
        return Feature(agg=agg, filters=filters, **data)


if __name__ == "__main__":
    ## test code
    sample_agg = Agg(
        columns=[Column(name="test1", data_type="int64")],
        logic="sum({0})",
        data_type="int64",
    )
    sample_filter = Filter(
        columns=[Column(name="test2", data_type="int64")],
        logic="{0} = 1",
    )
    sample_agg.query

    sample_feature = Feature(
        data_type="int64",
        topic="applprev_1",
        agg=sample_agg,
        filters=[sample_filter, sample_filter],
    )
    sample_feature.query
    sample_feature.name
