import gc
import json
import time
import polars as pl
from tqdm import tqdm
from dataset.feature.feature import *
from dataset.feature.util import optimize_dataframe 

from dataset.datainfo import RawInfo, RawReader, DATA_PATH
from dataset.feature.feature import *
from dataset.feature.util import optimize_dataframe
from dataset.const import TOPICS


topic = 'applprev'
type_ = 'train'
# load features from json file
with open(DATA_PATH / f'feature_definition/{topic}.json', 'r') as f:
    features = [Feature.from_dict(feature) for feature in json.load(f).values()]

rawinfo = RawInfo()
data = rawinfo.read_raw(topic, depth=1, reader=RawReader('polars'), type_=type_, stage='prep')
base = rawinfo.read_raw('base', reader=RawReader('polars'), type_=type_, stage='prep')
frame = data.join(base.select(['case_id', 'date_decision']), on='case_id', how='inner')


class FeatureBuilder:
    def __init__(self, frame: pl.DataFrame, features: List[Feature], batch_size: int = 500):
        self.frame = frame
        self.features = features
        self.batch_size = batch_size

    def execute_query(self, frame, features, batch_size):
        start_time = time.time()
        for i, index in enumerate(tqdm(range(0, len(features), batch_size))):  
            query = [
                f'cast({feat.query} as {feat.agg.data_type}) as {feat.name}'
                for feat in features[index : index + batch_size]
            ]
            temp = pl.SQLContext(frame=frame).execute(
                    f"""
                    SELECT frame.case_id
                        , {', '.join(query)}
                    from frame
                    group by frame.case_id"""
                    , eager=True
                )
            temp = optimize_dataframe(temp, verbose=True)
            temp.write_parquet(
                DATA_PATH / f'{type_}_feature/{type_}_{topic}_features_{i}.parquet',
            )
            del temp
            gc.collect()
        print(f'[*] Elapsed time: {time.time() - start_time:.4f} sec')


df = FeatureBuilder().execute_query(frame, features, 5000)
