from typing import List, Dict, Set

import yaml

from layers.transformation.embedding import Embedding
from layers.transformation.onehot import OneHot
from layers.transformation.binning import HistogramBinning
from layers.transformation.raw import Raw
from layers.transformation.standardization import Standardization
from layers.transformation.base_transformation import BaseTransformation
from parsing.feature.transformation_type import FeatureTransformation


class FeatureParser:
    def __init__(self):
        self.conf = {}

    def load_prop(self, features_file_path):
        with open(features_file_path) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            self.conf = self._parse(conf)

    def _get_required_fields_by_type(self, transformation_type: FeatureTransformation) -> Set[str]:
        if transformation_type == FeatureTransformation.EMBEDDING:
            return {'vocab_size', 'embedding_size'}
        elif transformation_type == FeatureTransformation.ONEHOT:
            return {'vocab_size'}
        elif transformation_type == FeatureTransformation.BINNING:
            return {'boundaries'}
        elif transformation_type == FeatureTransformation.STANDARDIZATION:
            return {'mean', 'stddev'}
        elif transformation_type == FeatureTransformation.RAW:
            return set()
        else:
            raise Exception(f'Wrong transformation type - {transformation_type}')


    def _parse_transformation(self,
                              transformation_type: FeatureTransformation,
                              feature_props: Dict[str, object],
                              num_hashing_bins: int) -> BaseTransformation:
        if transformation_type == FeatureTransformation.EMBEDDING:
            return Embedding(feature_props['vocab_size'], feature_props['embedding_size'], num_hashing_bins)
        elif transformation_type == FeatureTransformation.ONEHOT:
            return OneHot(feature_props['vocab_size'], num_hashing_bins)
        elif transformation_type == FeatureTransformation.BINNING:
            return HistogramBinning(feature_props['boundaries'], num_hashing_bins)
        elif transformation_type == FeatureTransformation.STANDARDIZATION:
            return Standardization(feature_props['mean'], feature_props['stddev'], num_hashing_bins)
        elif transformation_type == FeatureTransformation.RAW:
            return Raw(num_hashing_bins)
        else:
            raise Exception(f'Wrong transformation type - {transformation_type}')


    def _parse(self, conf: Dict[str, object]) -> Dict[str, BaseTransformation]:
        if 'features' not in conf:
            raise Exception('Please define "features".')

        transformation_by_feature = {}
        if 'transformations' in conf:
            for feature, props in conf['transformations'].items():
                if 'type' not in props:
                    raise Exception('Please define "type".')

                transformation_type = FeatureTransformation(props['type'])
                required_fields = self._get_required_fields_by_type(transformation_type)

                feature_props = {} if 'properties' not in props else props['properties']
                if not required_fields.issubset(feature_props.keys()):
                    raise Exception(f'Required fields for {feature}: {required_fields}')

                num_hashing_bins = 0 if 'num_hashing_bins' not in props else int(props['num_hashing_bins'])
                transformation = self._parse_transformation(transformation_type, feature_props, num_hashing_bins)
                transformation_by_feature[feature] = transformation

        return transformation_by_feature
