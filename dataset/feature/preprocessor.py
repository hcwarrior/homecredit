import gc
import shutil
import polars as pl
import os

from dataset.datainfo import RawInfo, RawReader, DATA_PATH
from dataset.feature.feature import *
from dataset.feature.util import optimize_dataframe
from dataset.const import TOPICS, DEPTH_2_TO_1_QUERY, CB_A_PREPREP_QUERY


class Preprocessor:

    def __init__(self, type_: str, conf: dict = None):
        self.raw_info = RawInfo(conf)
        self.type_ = type_

    def preprocess(self):
        for topic in TOPICS:
            gc.collect()
            if topic.depth <= 1 and topic.name not in DEPTH_2_TO_1_QUERY:
                print(f'[+] Memory optimization {topic.name}')
                self._memory_opt(topic.name, depth=topic.depth)
            elif topic.depth <= 1 and topic.name in DEPTH_2_TO_1_QUERY:
                # print(f'[+] skip {topic.name} because it is in DEPTH_2_TO_1_QUERY')
                pass
            elif topic.depth == 2 and topic.name in DEPTH_2_TO_1_QUERY:
                print(f'[+] Preprocessing {topic.name}, depth={topic.depth}')
                query = DEPTH_2_TO_1_QUERY[topic.name]
                if topic.name == 'credit_bureau_a':
                    self._preprocess_cb_a(topic.name, query)
                else:
                    self._preprocess_each(topic.name, query)
            elif topic.depth == 2 and topic.name not in DEPTH_2_TO_1_QUERY:
                raise ValueError(f'No query for {topic.name} in DEPTH_2_TO_1_QUERY but it is depth=2 topic')

    def _memory_opt(self, topic: str, depth: int):
        data = self.raw_info.read_raw(topic, depth=depth, reader=RawReader('polars'), type_=self.type_)
        data = optimize_dataframe(data)
        self.raw_info.save_as_prep(data, topic, depth=depth, type_=self.type_)

    def _join_depth2_0(self, depth1, depth2):
        depth2 = depth2.filter(pl.col('num_group2') == 0).drop('num_group2')
        depth1 = depth1.join(depth2, on=['case_id', 'num_group1'], how='left')
        return depth1

    def _preprocess_each(self, topic: str, query: str):
        depth2 = self.raw_info.read_raw(topic, depth=2, reader=RawReader('polars'), type_=self.type_)
        depth1 = self.raw_info.read_raw(topic, depth=1, reader=RawReader('polars'), type_=self.type_)
        temp = pl.SQLContext(data=depth2).execute(query, eager=True)
        depth1 = depth1.join(temp, on=['case_id', 'num_group1'], how='left')
        depth1 = self._join_depth2_0(depth1, depth2)

        depth1 = optimize_dataframe(depth1)
        self.raw_info.save_as_prep(depth1, topic, depth=1, type_=self.type_)

    def _preprocess_cb_a(self, topic: str, query: str):
        temp_path = DATA_PATH / 'parquet_preps' / self.type_
        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(temp_path/'agg', exist_ok=True)
        os.makedirs(temp_path/'depth2_0', exist_ok=True)

        iter = self.raw_info.read_raw_iter(topic, depth=2, reader=RawReader('polars'), type_=self.type_)
        for i, depth2 in enumerate(iter):
            depth2 = optimize_dataframe(depth2)

            depth2_0 = depth2.filter(pl.col('num_group2') == 0).drop('num_group2')
            temp_file = temp_path / 'depth2_0'/ f"{self.type_}_{topic}_1_temp_{i}.parquet"
            depth2_0.write_parquet(temp_file)
            del depth2_0

            depth2 = pl.SQLContext(data=depth2).execute(
                CB_A_PREPREP_QUERY,
                eager=True,
            )
            depth2 = optimize_dataframe(depth2)
            depth2 = pl.SQLContext(data=depth2).execute(query, eager=True)
            depth2 = optimize_dataframe(depth2)
            temp_file = temp_path / 'agg' / f"{self.type_}_{topic}_1_temp_{i}.parquet"
            depth2.write_parquet(temp_file)
            del depth2
            gc.collect()            

        depth1 = self.raw_info.read_raw(topic, depth=1, reader=RawReader('polars'), type_=self.type_)
        depth1 = optimize_dataframe(depth1)

        files = [f for f in os.listdir(temp_path / 'agg') if f.endswith('.parquet')]
        dfs = [pl.read_parquet(temp_path/'agg'/file) for file in files]
        depth2_temp = pl.concat(dfs, how='vertical_relaxed')
        depth1 = depth1.join(depth2_temp, on=['case_id', 'num_group1'], how='left')
        del depth2_temp
        gc.collect()

        files = [f for f in os.listdir(temp_path / 'depth2_0') if f.endswith('.parquet')]
        dfs = [pl.read_parquet(temp_path / 'depth2_0' / file) for file in files]
        depth2_temp = pl.concat(dfs, how='vertical_relaxed')
        depth1 = depth1.join(depth2_temp, on=['case_id', 'num_group1'], how='left')
        del depth2_temp
        gc.collect()

        self.raw_info.save_as_prep(depth1, topic, depth=1, type_=self.type_)

        # remove temp files
        shutil.rmtree(temp_path / 'agg')
        shutil.rmtree(temp_path / 'depth2_0')

if __name__ == "__main__":
    prep = Preprocessor('train')
    prep.preprocess()
