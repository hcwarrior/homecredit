import gc
import json
import time
import polars as pl
from tqdm import tqdm
from dataset.feature.feature import *
from dataset.feature.util import optimize_dataframe 

# preprocess

from dataset import const

from dataset.datainfo import RawInfo, RawReader
from dataset.feature.feature import *
from dataset.const import TOPICS
from typing import Dict, List
import json


raw_info = RawInfo()
df = RawInfo.read_raw('applprev', depth=2, reader=RawReader(return_type='polars'))
df


# const.TOPICS


# load features from json file
with open('data/feature_definition/applprev.json', 'r') as f:
    features = [Feature.from_dict(feature) for feature in json.load(f).values()]


data = pl.read_parquet(
    'data/home-credit-credit-risk-model-stability/parquet_files/train/train_applprev_1_0.parquet',
)
base = pl.read_parquet(
    'data/home-credit-credit-risk-model-stability/parquet_files/train/train_base.parquet',
)
frame = data.join(base.select(['case_id', 'date_decision']), on='case_id', how='inner')


class FeatureBuilder:
    def __init__(self, frame: pl.DataFrame, features: List[Feature], batch_size: int = 500):
        self.frame = frame
        self.features = features
        self.batch_size = batch_size

    def execute_query(self):
        start_time = time.time()
        for i, index in enumerate(tqdm(range(0, len(self.features), self.batch_size))):  
            query = [
                f'cast({feat.query} as {feat.agg.data_type}) as {feat.name}'
                for feat in self.features[index : index + self.batch_size]
            ]
            temp = pl.SQLContext(frame=self.frame).execute(
                    f"""
                    SELECT frame.case_id
                        , {', '.join(query)}
                    from frame
                    group by frame.case_id"""
                    , eager=True
                )
            print('[*] Optimizing dataframe')
            temp = optimize_dataframe(temp, verbose=True)
            temp.write_parquet(
                f'data/home-credit-credit-risk-model-stability/parquet_files/train_feature/train_applprev_1_0_features_{i}.parquet',
            )
            del temp
            gc.collect()
        print(f'[*] Elapsed time: {time.time() - start_time:.4f} sec')

df = FeatureBuilder(frame, features, 5000, 10).execute_query()
