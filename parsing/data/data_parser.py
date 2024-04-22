import glob
import itertools
from typing import Dict, Iterator, Tuple, List

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype


class DatasetGenerator:
    _NA_STRING_VAL = 'NA'
    _SUPPORTED_EXTENSIONS = ['parquet', 'csv']
    def __init__(self, root_dir: str, input_features: List[str], target: str, id: str):
        self.root_dir = root_dir
        self.files = list(itertools.chain.from_iterable(
            [glob.glob(f'{root_dir}/*.{ext}') for ext in DatasetGenerator._SUPPORTED_EXTENSIONS]))
        self.label = target
        self.features = input_features + [target, id]

    # returns numpy array dict
    def parse(self) -> Iterator[Tuple[str, Dict[str, np.ndarray]]]:
        file_iterator = itertools.cycle(self.files)
        for file_path in file_iterator:
            yield file_path, self._to_numpy_array_dict(self._parse_file_to_frame(file_path), self.label)

    def _parse_file_to_frame(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith('csv'):
            df = pd.read_csv(file_path)[self.features]
        elif file_path.endswith('parquet'):
            df = pd.read_parquet(file_path, columns=self.features, engine='fastparquet')
        else:
            raise Exception(f'Unsupported file - {file_path}')

        self._fill_na(df)
        return df

    def _fill_na(self, df: pd.DataFrame):
        for col in df.columns:
            if is_string_dtype(df[col].dtype):
                df.fillna({col: DatasetGenerator._NA_STRING_VAL}, inplace=True)
            else:
                df[col] = df[col].bfill()

    def _to_numpy_array_dict(self, df: pd.DataFrame, label: str) -> Dict[str, np.ndarray]:
        df[label] = df[label].astype('int32')
        return {col: df[col].values for col in self.features}

