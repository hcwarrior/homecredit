import os
import pandas as pd
from pathlib import Path
from argparse import Namespace
from dataclasses import dataclass
from fastparquet import ParquetFile


BASE_PATH = Path(os.getcwd())
DATA_PATH = BASE_PATH / "data" / "home-credit-credit-risk-model-stability"
POSTFIXES = {
    "P": "Transform DPD (Days Past Due)",
    "M": "Masking Categories",
    "A": "Transform Amount",
    "D": "Transform Date",
    "T": "Unspecified Transform",
    "L": "Unspecified Transform",
}


class RawFile:
    def __init__(self, file_name: str = "") -> None:
        self.file_name = str(file_name)

        if isinstance(self.file_name, str) and self.file_name:
            (
                self._type,
                self._name,
                self._depth,
                self._index,
                self._file_format
            ) = self._parse_file_name()
        else:
            raise ValueError(f"file_name should be a non-empty string. Not {file_name}.")

    def __repr__(self) -> str:
        return f"{self.file_name}"

    def __str__(self) -> str:
        return self.file_name

    def __lt__(self, other) -> bool:
        return self.file_name < other.file_name

    @property
    def type(self) -> str:
        return self._type
    
    @property
    def depth(self) -> str:
        return self._depth
    
    @property
    def index(self) -> str:
        return self._index
    
    @property
    def format(self) -> str:
        return self._file_format
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def fullname(self) -> str:
        return self.file_name.rsplit(".", 1)[0]

    def _parse_file_name(self) -> tuple[str, str, str, str]:
        fullname = self.fullname
        file_format = self.file_name.rsplit(".", 1)[1]
        
        names = fullname.split("_")
        if names[-2].isdigit():
            return names[0], "_".join(names[1:-2]), names[-2], names[-1], file_format
        elif names[-1].isdigit():
            return names[0], "_".join(names[1:-1]), names[-1], "", file_format
        else:
            return names[0], "_".join(names[1:]), "", "", file_format

    def get_path(self, data_dir: Path = None) -> Path:
        if data_dir is None:
            data_dir = DATA_PATH
        return Path(data_dir) / f"{self.format}_files" / self.type / self.file_name
    
    def startswith(self, keyword: str) -> bool:
        return self.file_name.startswith(keyword)


@dataclass
class ColInfo:
    name: str

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    @property
    def desc(self) -> str:
        return self.describe()

    def describe(
            self,
            description_file: str = "feature_definitions.csv"
    ) -> str:
        description_df = pd.read_csv(DATA_PATH / description_file, usecols=["Variable", "Description"])

        if self.name in description_df["Variable"].values:
            result_description = (
                description_df.loc[
                    description_df["Variable"] == self.name, "Description"
                ].values[0])
        else:
            result_description = self.name

        if self.name[-1] in POSTFIXES:
            return f"{self.name}: {result_description} ({POSTFIXES[self.name[-1]]})"
        else:
            return f"{self.name}: {result_description}"


class RawInfo:
    VALID_TYPES = ["", "train", "test"]
    VALID_DEPTHS = ["", "0", "1", "2"]

    def __init__(self, config: dict = None, format_: str = "parquet") -> None:
        self.config = config
        if self.config is None:
            self.config = Namespace(**{
                "data_path": DATA_PATH,
            })

        self.format = format_
        self.data_dir_path = Path(self.config.data_path)
        self.file_dir_path = self.data_dir_path / f"{self.format}_files"

        if not self.file_dir_path.exists():
            raise FileNotFoundError(f"{self.file_dir_path} does not exist.")
        
        self.reader = RawReader(self.format)

    def show_files(self, type_: str = "train") -> list[RawFile]:
        return sorted([RawFile(f) for f in os.listdir(self.file_dir_path / type_)])

    def get_files(self, filename: str, *, depth: int = None, type_: str = "train") -> list[RawFile]:
        if depth is None:
            return sorted([
                f for f in self.show_files(type_) if f.name == filename])
        else:
            return sorted([
                f for f in self.show_files(type_)
                if f.name == filename and f.depth == str(depth)])

    def get_depths_by_name(self, file_name: str, type_: str = "train") -> list[int]:
        return sorted(list(set([int(f.depth) for f in self.get_files(file_name, type_=type_)])))

    def get_files_by_depth(self, depth: int, type_: str = "train") -> list[RawFile]:
        return [f for f in self.show_files(type_) if f.depth == str(depth)]

    def read_raw(
        self,
        file_name: str,
        *,
        depth: int = None,
        type_: str = "train",
    ) -> pd.DataFrame:
        raw_files = self.get_files(file_name, depth=depth, type_=type_)

        if len(raw_files) > 0:
            raw_df = pd.concat([self.reader(rf.get_path(self.data_dir_path)) for rf in raw_files])
        else:
            raise FileNotFoundError(f"{file_name} (depth: {depth}) does not exist in {type_} files.")

        return raw_df


class RawReader:
    def __init__(self, format_: str = "parquet") -> None:
        self.format = format_
        
        if format_ == "parquet":
            self.reader = pd.read_parquet
            self.column_getter = self._get_parquet_columns
        elif format_ == "csv":
            self.reader = pd.read_csv
            self.column_getter = self._get_csv_columns
        else:
            raise ValueError(f"format_ should be either 'parquet' or 'csv'. Not {format_}.")

    def read(self, file_path: Path) -> pd.DataFrame:
        return self.reader(file_path)

    def columns(self, file_path: Path) -> list[ColInfo]:
        return [ColInfo(c) for c in self.column_getter(file_path)]

    def _get_csv_columns(self, file_path: Path) -> list[ColInfo]:
        return [c for c in self.reader(file_path, nrows=0).columns]

    def _get_parquet_columns(self, file_path: Path) -> list[ColInfo]:
        return [c for c in ParquetFile(file_path).columns]

    def __call__(self, file_path) -> pd.DataFrame:
        return self.read(file_path)


if __name__ == "__main__":
    raw_info = RawInfo(
        Namespace(
            data_path = 
                "/Users/jaehwi/Projects/kaggle/homecredit/data/home-credit-credit-risk-model-stability"
        ))
    print(raw_info.show_files())
    print(raw_info.get_files("base"))
    print(raw_info.get_files_by_depth(0))
    print(raw_info.get_files("static"))
    print(raw_info.get_files("applprev"))
    print(raw_info.get_files("applprev", 1))
    print(raw_info.get_files("applprev", 2))
    print(raw_info.read_raw("static", depth=0).head())

    test_reader = RawReader("parquet")
    column_infos = test_reader.columns(raw_info.get_files("static")[0].get_path())
    print(column_infos)
    print(column_infos[1].desc)