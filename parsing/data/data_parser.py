import glob

import pandas as pd


class DatasetGenerator:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.files = glob.glob(root_dir)

    def _parse_file_to_frame(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith('csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('parquet'):
            return pd.read_parquet(file_path, engine='fastparquet')
        else:
            print(f'Unsupported file - {file_path}')

    # returns a pandas dataframe
    def __iter__(self):
        for file_path in self.files:
            yield self._parse_file_to_frame(file_path)
