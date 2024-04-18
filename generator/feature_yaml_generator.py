from typing import Dict, Set

import numpy as np
import pandas as pd
import yaml

class FeatureYAMLGenerator:
    def __init__(self,
                 feature_conf_yaml_path: str,
                 parquet_data_file_path: str,
                 output_yaml_path: str):
        self.parquet_data_file_path = parquet_data_file_path
        self.output_yaml_path = output_yaml_path
        with open(feature_conf_yaml_path) as f:
            self.conf = yaml.load(f, Loader=yaml.FullLoader)

    def generate(self):
        props = self._parse(self.conf)
        with open(self.output_yaml_path, 'w') as f:
            yaml.dump(props, f)

    def _parse(self, conf: Dict[str, object]) -> Dict[str, object]:
        features = conf['features']
        continuous_features = features['continuous']
        categorical_features = features['categorical']

        props = {}
        for feature in features:
            prop = self._generate_prop(feature, continuous_features, categorical_features)
            props[feature] = prop

        return props

    def _generate_prop(self,
                       column: str,
                       continuous_features: Set[str],
                       categorical_features: Set[str]) -> Dict[str, object]:
        if not self.parquet_data_file_path.endswith('parquet'):
            raise Exception(f'Unsupported file - {self.parquet_data_file_path}')

        # read only single column to be memory-efficient
        df = pd.read_parquet(self.parquet_data_file_path, columns=[column], engine='fastparquet')
        series = df[column].dropna()
        prop = {}
        if column in continuous_features:
            if abs(series.skew()) >= 1.0:
                boundaries = series.quantile(np.arange(0.05, 1.0, 0.05))
                prop['type'] = 'binning'
                prop['properties'] = {'boundaries': boundaries}
            else:
                mean, stddev = series.mean(), series.std()
                prop['type'] = 'standardization'
                prop['properties'] = {'mean': mean, 'stddev': stddev}
        else: # elif column in categorical_features
            # num_uniques = series.nunique()
            # if num_uniques >= 10:
            #     prop['type'] = 'character_embedding'
            #     prop['properties'] = {'vocab_size': num_uniques, 'embedding_size': num_uniques // 3}
            # else:
            prop['type'] = 'onehot'
            prop['properties'] = {'vocab': series.unique().tolist()}


        return prop