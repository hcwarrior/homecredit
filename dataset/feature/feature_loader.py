import gc
import json
import os
import time
import polars as pl
from tqdm import tqdm
from dataset.feature.feature import *
from dataset.feature.feature_definer import FEATURE_DEF_PATH
from dataset.feature.feature import *
from dataset.feature.util import optimize_dataframe

from dataset.datainfo import RawInfo, RawReader, DATA_PATH
from dataset.const import TOPICS, Topic, KEY_COL, DATE_COL, TARGET_COL


class FeatureLoader:
    def __init__(self, topic: Topic, type: str):
        self.topic = topic
        self.type = type
        self.data = self._load_data(type_=type, stage='prep')

    def _load_data(
        self,
        type_='train',
        stage='prep',
        rawinfo=RawInfo(),
        reader=RawReader('polars'),
    ) -> pl.DataFrame:
        base_columns = [*KEY_COL, *DATE_COL]
        if type_ == 'train':
            base_columns += TARGET_COL
        data = rawinfo.read_raw(
            self.topic.name,
            depth=self.topic.depth,
            reader=reader,
            type_=type_,
            stage=stage,
        )
        base = rawinfo.read_raw('base', reader=reader, type_=type_)
        base = base.with_columns(pl.col(KEY_COL).cast(pl.Int32))
        return data.join(base.select(base_columns), on=KEY_COL, how='inner')

    def load_features(self, feature_names: List[str] = None) -> List[Feature]:
        if not os.path.exists(FEATURE_DEF_PATH / f'{self.topic.name}.json'):
            raise FileNotFoundError(
                f'Feature definition for {self.topic.name} not found.'
            )

        with open(FEATURE_DEF_PATH / f'{self.topic.name}.json', 'r') as f:
            features = json.load(f)

        if feature_names is None:
            return [Feature.from_dict(feature) for feature in features.values()]
        return [Feature.from_dict(features[feature_name]) for feature_name in feature_names]

    def load_feature_data(self, features, verbose=False) -> pl.DataFrame:
        query = [
            f'cast({feat.query} as {feat.agg.data_type}) as {feat.name}'
            for feat in features
        ]
        if verbose:
            for q in query:
                print(f'[*] Query: {q}')
        temp = pl.SQLContext(frame=self.data).execute(
            f"""
            SELECT frame.case_id, frame.target
                , {', '.join(query)}
            from frame
            group by frame.case_id, frame.target
            """.replace(
                'float32', 'float'
            )
            .replace(
                "min_pmts_year_1139T507T__D",
                "case when min_pmts_year_1139T507T__D='-01-01' then null else replace(min_pmts_year_1139T507T__D, '.0', '') end",
            )
            .replace(
                "max_pmts_year_1139T507T__D",
                "case when max_pmts_year_1139T507T__D='-01-01' then null else replace(max_pmts_year_1139T507T__D, '.0', '') end",
            ),
            eager=True,
        )
        temp = optimize_dataframe(temp)
        return temp

    def load_feature_data_batch(self, features, batch_size, verbose=False, skip=0):
        """
        Load feature data in batch
        """
        start_time = time.time()
        for i, index in enumerate(tqdm(range(0, len(features), batch_size))):
            if i < skip:
                yield None
            else:
                yield self.load_feature_data(
                    features[index : index + batch_size], verbose=verbose
                )
        print(f'[*] Elapsed time: {time.time() - start_time:.4f} sec')

    # def read_df(self, path: str, feature_names: List[str]=None) -> pl.DataFrame:
    #     df = pl.read_parquet(path)
    #     if feature_names is None:
    #         return df
    #     if self.type != 'train':
    #         feature_names += [KEY_COL]
    #     else:
    #         feature_names += [KEY_COL, TARGET_COL]
    #     return df.select([c for c in df.columns if c in feature_names])
