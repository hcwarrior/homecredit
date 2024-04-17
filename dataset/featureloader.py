import pandas as pd
import os
from pathlib import Path
from glob import glob
from dataclasses import dataclass


@dataclass
class FeatureInfo:
    topic: str
    feature_name: str

    def __post_init__(self):
        self.validate_topic()

    def validate_topic(self):
        if self.topic not in self.TOPICS:
            raise ValueError(f"Invalid topic: {self.topic}. Available topics: {self.TOPICS}")


class FeatureLoader:
    DATA_KEYS = ["case_id"]
    TOPICS = [
        "applprev_1",
        "applprev_2",
        "credit_bureau_a_1",
        "credit_bureau_a_2",
        "credit_bureau_b_1",
        "credit_bureau_b_2",
        "debitcard_1",
        "deposit_1",
        "other_1",
        "person_1",
        "person_2",
        "static_0",
        "static_cb",
        "tax_registry_a",
        "tax_registry_b",
        "tax_registry_c",
    ]
    BASE_PATH = Path(os.getcwd())
    DATA_PATH = BASE_PATH / "data" / "home-credit-credit-risk-model-stability"
    PARQUET_DIR_PATH = DATA_PATH / "parquet_files"

    def __init__(self, feature_path):
        self.feature_path = feature_path

    def load(self):
        data = pd.read_csv(self.data_path)
        feature = pd.read_csv(self.feature_path)
        return data, feature

    def read_data(
        file_name: str,
        depth: int,
        type_: str = "train",
        dir_path: str = PARQUET_DIR_PATH,
        format_: str = "parquet",
    ):
        file_dir = os.path.join(dir_path, type_)
        file_name_format = f"{file_name}_{depth}*.{format_}"

        if file_name in ["train_base", "test_base"]:
            file_path = os.path.join(file_dir, f"{file_name}.{format_}")
            return pd.read_parquet(file_path)

        files = [f for f in glob.glob(os.path.join(file_dir, file_name_format))]
        if len(files) == 0:
            raise FileNotFoundError(
                f"No file found with the name '{file_name_format}' in '{file_dir}'"
            )
        return pd.read_parquet(files)
