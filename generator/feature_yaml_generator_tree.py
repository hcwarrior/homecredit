from typing import Dict, Set

import numpy as np
import pandas as pd
import yaml

class FeatureYAMLGeneratorTree:
    def __init__(self,
                 feature_conf_yaml_path: str,
                 parquet_data_root_dir: str,
                 output_yaml_path: str):
        self.parquet_data_root_dir = parquet_data_root_dir
        self.output_yaml_path = output_yaml_path
        with open(feature_conf_yaml_path) as f:
            self.conf = yaml.load(f, Loader=yaml.FullLoader)

    def generate(self):
        props = self._parse(self.conf)
        with open(self.output_yaml_path, 'w') as f:
            yaml.dump(props, f, default_flow_style=None)

    def _parse(self, conf: Dict[str, object]) -> Dict[str, object]:
        features = conf['features']
        continuous_features = features['continuous']
        categorical_features = features['categorical']

        all_features = set(continuous_features + categorical_features)

        props = {}
        for feature in all_features:
            prop = self._generate_prop(feature, features['label'], set(continuous_features), set(categorical_features))
            props[feature] = prop

        return {'transformations': props}

    def _generate_prop(self,
                       column: str,
                       label: str,
                       continuous_features: Set[str],
                       categorical_features: Set[str]) -> Dict[str, object]:
        # read only single column to be memory-efficient
        df = pd.read_parquet(self.parquet_data_root_dir, columns=[column, label], engine='pyarrow')
        series_with_label = df.dropna()
        prop = {}
        if column in continuous_features:
            series = series_with_label[column]
            boundaries = series.quantile(np.arange(0.1, 1.0, 0.1))
            if boundaries.nunique() >= 9 and abs(series.skew()) >= 2.0:
                prop['type'] = 'binning'
                prop['properties'] = {'boundaries': boundaries.tolist()}
            else:
                mean, stddev = float(series.mean()), float(series.std())
                prop['type'] = 'standardization'
                prop['properties'] = {'mean': mean, 'stddev': stddev}
        else: # elif column in categorical_features
            series = series_with_label[column]
            uniques = list(series.unique()) + ['NA']
            if len(uniques) >= 10:
                prop['type'] = 'target_encoding'

                target_encoded = series_with_label.groupby(column, observed=False)[label].mean()
                target_encoded['NA'] = 0
                encoded_df = pd.DataFrame({'value': target_encoded.index.values, 'encoded': target_encoded.values})

                values, encoded = [], []
                for _, row in encoded_df.iterrows():
                    values.append(row['value'])
                    encoded.append(row['encoded'])
                prop['properties'] = {'value': values, 'encoded': encoded}
            else:
                prop['type'] = 'onehot'
                prop['properties'] = {'vocab': uniques}

        return prop