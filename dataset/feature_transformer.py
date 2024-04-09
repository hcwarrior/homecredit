from typing import List, Union

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class GroupbyTransform(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        agg: str,
        groupby: Union[str, List[str]],
        column: Union[str, List[str]],
    ) -> None:
        if agg not in ["count", "min", "max"]:
            raise ValueError("agg must be 'count', 'min' or 'max'")

        self.agg = agg
        self.groupby = groupby
        self.column = column
        self.postfix = f"_gb{agg}"

    def fit(self, X: pd.DataFrame = None, y: pd.Series = None) -> BaseEstimator:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.column, str):
            X[f"{self.column}{self.postfix}"] = self._transform(X)
        elif isinstance(self.column, list):
            columns = [f"{col}{self.postfix}" for col in self.column]
            X[columns] = self._transform(X)
        else:
            raise ValueError("column must be str or List[str]")
        return X

    def _transform(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return X.groupby(self.groupby)[self.column].transform(self.agg)


class GroupbyCount(GroupbyTransform):
    def __init__(
        self,
        groupby: Union[str, List[str]],
        column: Union[str, List[str]],
    ) -> None:
        super().__init__("count", groupby, column)


class GroupbyMin(GroupbyTransform):
    def __init__(
        self,
        groupby: Union[str, List[str]],
        column: Union[str, List[str]],
    ) -> None:
        super().__init__("min", groupby, column)


class GroupbyMax(GroupbyTransform):
    def __init__(
        self,
        groupby: Union[str, List[str]],
        column: Union[str, List[str]],
    ) -> None:
        super().__init__("max", groupby, column)


class ToDatetime(BaseEstimator, TransformerMixin):
    def __init__(self, column: Union[str, List[str]], format: str = "%Y-%m-%d") -> None:
        self.column = column
        self.format = format
        self.postfix = "_dt"

    def fit(self, X: pd.DataFrame = None, y: pd.Series = None) -> BaseEstimator:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform(X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.column, str):
            X[f"{self.column}{self.postfix}"] = pd.to_datetime(X[self.column], format=self.format)
        elif isinstance(self.column, list):
            columns = [f"{col}{self.postfix}" for col in self.column]
            X[columns] = pd.to_datetime(X[self.column], format=self.format)
        else:
            raise ValueError("column must be str or List[str]")
        return X
