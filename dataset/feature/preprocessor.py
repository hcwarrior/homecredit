import polars as pl
import os

from dataset.datainfo import RawInfo, RawReader, DATA_PATH
from dataset.feature.feature import *
from dataset.feature.util import optimize_dataframe
from dataset.const import TOPICS, DEPTH_2_TO_1_QUERY, CB_A_PREPREP_QUERY


class Preprocessor:
    RAW_INFO = RawInfo()

    def __init__(self, type_: str):
        self.type_ = type_

    def preprocess(self):
        for topic in TOPICS:
            print(f'\n[*] Preprocessing {topic.name}, depth={topic.depth}')
            if topic.depth <= 1 and topic.name not in DEPTH_2_TO_1_QUERY:
                print(f'  [+] Memory optimization {topic.name}')
                self._memory_opt(topic.name, depth=topic.depth)
            elif topic.depth <= 1 and topic.name in DEPTH_2_TO_1_QUERY:
                print(f'  [+] skip {topic.name} because it is in DEPTH_2_TO_1_QUERY')
            elif topic.depth == 2 and topic.name in DEPTH_2_TO_1_QUERY:
                query = DEPTH_2_TO_1_QUERY[topic.name]
                if topic.name == 'credit_bureau_a':
                    self._preprocess_cb_a(topic.name, query)
                    print(f'  [+++] skip {topic.name} because it is credit_bureau_a')
                else:
                    self._preprocess_each(topic.name, query)
            elif topic.depth == 2 and topic.name not in DEPTH_2_TO_1_QUERY:
                raise ValueError(f'No query for {topic.name} in DEPTH_2_TO_1_QUERY but it is depth=2 topic')

    def _memory_opt(self, topic: str, depth: int):
        data = self.RAW_INFO.read_raw(topic, depth=depth, reader=RawReader('polars'), type_=self.type_)
        data = optimize_dataframe(data, verbose=True)
        self.RAW_INFO.save_as_prep(data, topic, depth=depth, type_=self.type_)

    def _join_depth2_0(self, depth1, depth2):
        depth2 = depth2.filter(pl.col('num_group2') == 0).drop('num_group2')
        depth1 = depth1.join(depth2, on=['case_id', 'num_group1'], how='left')
        return depth1

    def _preprocess_each(self, topic: str, query: str):
        depth2 = self.RAW_INFO.read_raw(topic, depth=2, reader=RawReader('polars'), type_=self.type_)
        depth1 = self.RAW_INFO.read_raw(topic, depth=1, reader=RawReader('polars'), type_=self.type_)
        temp = pl.SQLContext(data=depth2).execute(query, eager=True)
        depth1 = depth1.join(temp, on=['case_id', 'num_group1'], how='left')
        depth1 = self._join_depth2_0(depth1, depth2)

        depth1 = optimize_dataframe(depth1, verbose=True)
        self.RAW_INFO.save_as_prep(depth1, topic, depth=1, type_=self.type_)

    def _preprocess_cb_a(self, topic: str, query: str):
        depth2 = self.RAW_INFO.read_raw(topic, depth=2, reader=RawReader('polars'), type_=self.type_)
        print('[*] Read depth=2 data')
        depth2 = optimize_dataframe(depth2, verbose=True)

        depth1 = self.RAW_INFO.read_raw(topic, depth=1, reader=RawReader('polars'), type_=self.type_)
        print('[*] Read depth=1 data')
        depth1 = optimize_dataframe(depth1, verbose=True)
        depth1 = self._join_depth2_0(depth1, depth2)

        temp_path = (
            DATA_PATH / 'parquet_preps' / self.type_ / f"{self.type_}_{topic}_temp.parquet"
        )
        print(f'[*] Temp path: {temp_path}')
        if not os.path.exists(temp_path):
            depth2_temp = pl.SQLContext(data=depth2).execute(
                CB_A_PREPREP_QUERY,
                eager=True,
            )
            depth2_temp.write_parquet(temp_path)
            print('[*] Temp data saved')
        else:
            depth2_temp = pl.read_parquet(temp_path)
            print('[*] Read temp data')
        depth2_temp = pl.SQLContext(data=depth2_temp).execute(query, eager=True)
        depth2_temp = optimize_dataframe(depth2_temp, verbose=True)
        depth1 = depth1.join(depth2_temp, on=['case_id', 'num_group1'], how='left')

        self.RAW_INFO.save_as_prep(depth1, topic, depth=1, type_=self.type_)

if __name__ == "__main__":
    prep = Preprocessor('train')
    prep.preprocess()


# fd = FeatureDefiner('person', period_cols=None, depth=2)
# features = fd.define_simple_features(20)
# for feature in features:
#     print(f', {feature.query} as {feature.name}')

# fd = FeatureDefiner('applprev', period_cols=None, depth=2)
# features = fd.define_simple_features(20)
# for feature in features:
#     print(f', {feature.query} as {feature.name}')

# fd = FeatureDefiner('credit_bureau_a', period_cols=None, depth=2)
# features = fd.define_simple_features(20)
# for feature in features:
#     print(f', {feature.query} as {feature.name}')

# fd = FeatureDefiner('credit_bureau_b', period_cols=None, depth=2)
# features = fd.define_simple_features(20)
# for feature in features:
#     print(f', {feature.query} as {feature.name}')
