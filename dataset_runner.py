import gc
import json
import time
import polars as pl
from tqdm import tqdm
from dataset.feature.feature import *
from dataset.feature.util import optimize_dataframe 

# preprocess
from dataset.datainfo import RawInfo, RawReader
from dataset.feature.feature import *
from dataset.const import TOPICS
from typing import Dict, List

rawinfo = RawInfo()

depth1 = rawinfo.read_raw('applprev', depth=1, reader=RawReader('polars'))
depth2 = rawinfo.read_raw('applprev', depth=2, reader=RawReader('polars'))
temp = pl.SQLContext(data=depth2).execute(
    f"""
        SELECT case_id, num_group1
            , count(cacccardblochreas_147M) as cacccardblochreas_147M_cntL
            , count(conts_type_509L) as conts_type_509L_cntL
            , count(credacc_cards_status_52L) as credacc_cards_status_52L_cntL
        from data
        group by case_id, num_group1
        """,
    eager=True,
)
depth1 = depth1.join(temp, on=['case_id', 'num_group1'], how='left')

depth2 = depth2.filter(pl.col('num_group2') == 0).drop('num_group2')
depth1 = depth1.join(depth2, on=['case_id', 'num_group1'], how='left')

depth1 = optimize_dataframe(depth1, verbose=True)
rawinfo.save_as_prep(depth1, 'applprev', depth=1)

# addres_role_871L
# conts_role_79M
# onehot??

depth2 = rawinfo.read_raw('person', depth=2)

addres_role_871L
empls_economicalst_849M
relatedpersons_role_762T

for col in depth2.columns:
    print(f'{col}: {depth2[col].nunique()}')
depth2.relatedpersons_role_762T.value_counts()

depth1 = rawinfo.read_raw('person', depth=1, reader=RawReader('polars'))
depth2 = rawinfo.read_raw('person', depth=2, reader=RawReader('polars'))
temp = pl.SQLContext(data=depth2).execute(
    f"""
        SELECT case_id, num_group1
            , count(cacccardblochreas_147M) as cacccardblochreas_147M_cntL
            , count(conts_type_509L) as conts_type_509L_cntL
            , count(credacc_cards_status_52L) as credacc_cards_status_52L_cntL
        from data
        group by case_id, num_group1
        """,
    eager=True,
)
depth1 = depth1.join(temp, on=['case_id', 'num_group1'], how='left')

depth2 = depth2.filter(pl.col('num_group2') == 0).drop('num_group2')
depth1 = depth1.join(depth2, on=['case_id', 'num_group1'], how='left')

depth1 = optimize_dataframe(depth1, verbose=True)
rawinfo.save_as_prep(depth1, 'applprev', depth=1)

#
#
#
#
#
#
#
#
#
#
# load features from json file
with open('data/feature_definition/applprev.json', 'r') as f:
    features = [Feature.from_dict(feature) for feature in json.load(f).values()]


data = rawinfo.read_raw('applprev', depth=1, reader=RawReader('polars'))
base = rawinfo.read_raw('base', reader=RawReader('polars'))

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
