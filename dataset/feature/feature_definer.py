import pandas as pd
import os
from pathlib import Path
from glob import glob
from dataclasses import dataclass
from const import TOPICS
from typing import List, Union

class FeatureDefiner:
    def __init__(self,
                 topic: str,
                 key_columns: List[str],
                 agg_columns: List[str],
                 filter_columns: List[str],
        ):
        self.topic = topic
        self.key_columns = key_columns
        self.agg_columns = agg_columns
        self.filter_columns = filter_columns

        if not self.key_columns:
            self.key_columns = ["case_id"]
        if not self.agg_columns:
            self.agg_columns = []
        if not self.filter_columns:
            self.filter_columns = []
        
        self.agg_components = self._get_agg_components()
        self.filter_components = self._get_filter_components()
        # preset of agg (sum, mean, etc.)
        self.agg_preset = [
            "sum",
            "mean",
            "std",
            "min",
            "max",
            "count",
        ]

    def update_compoment():
        self.agg_components = self._get_agg_components()
        self.filter_components = self._get_filter_components()
    
    def _get_agg_components(self) -> List[str]:
        return [f"{col}_agg" for col in self.agg_columns]
    
    def _get_filter_components(self) -> List[str]:
        return [f"{col}_filter" for col in self.filter_columns]
    
    def get_agg_data(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        lf_agg = lf.groupby(self.key_columns).agg(self.agg_components)
        return lf_agg
    
    def get_filter_data(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        lf_filter = lf.filter(self.filter_components)
        return lf_filter
    
    def define_feature(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        lf_agg = self.get_agg_data(lf)
        lf_filter = self.get_filter_data(lf)
        return lf_agg.join(lf_filter)

