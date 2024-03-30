import glob
from typing import Dict

import numpy as np
import pandas as pd


class DatasetGenerator:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.files = glob.glob(f'{root_dir}/*')

    # returns numpy array dict
    def parse(self):
        for file_path in self.files:
            print(file_path)
            yield self._to_numpy_array_dict(self._parse_file_to_frame(file_path))

    def _parse_file_to_frame(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith('csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('parquet'):
            return pd.read_parquet(file_path, engine='fastparquet')
        else:
            print(f'Unsupported file - {file_path}')

    def _to_numpy_array_dict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {col: df[col].values for col in df.columns}

