from typing import List, Dict, Set

import yaml

from layers.transformation.embedding import Embedding
from layers.transformation.onehot import OneHot
from layers.transformation.binning import HistogramBinning
from layers.transformation.standardization import Standardization
from layers.transformation.base_transformation import BaseTransformation
from transformation_type import FeatureTransformation


class FeatureParser:
    _TYPE_NAMES: List[str] = [x.value.name for x in FeatureTransformation]

    def __init__(self, features_file_path: str):
        self.transformations_by_feature = self._load_prop(features_file_path)

    def _load_prop(self, features_file_path) -> Dict[str, BaseTransformation]:
        with open(features_file_path) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            return self._parse(conf)

    def _get_required_fields_by_type(self, transformation_type: FeatureTransformation) -> Set[str]:
        if transformation_type == FeatureTransformation.EMBEDDING:
            return {'vocab_size', 'embedding_size'}
        elif transformation_type == FeatureTransformation.ONEHOT:
            return {'vocab_size'}
        elif transformation_type == FeatureTransformation.BINNING:
            return {'boundaries'}
        elif transformation_type == FeatureTransformation.STANDARDIZATION:
            return {'mean', 'stddev'}
        else:
            raise Exception(f'Wrong transformation type - {transformation_type}')


    def _parse_transformation(self,
                              transformation_type: FeatureTransformation,
                              feature_props: Dict[str, object]) -> BaseTransformation:
        if transformation_type == FeatureTransformation.EMBEDDING:
            return Embedding(feature_props['vocab_size'], feature_props['embedding_size'])
        elif transformation_type == FeatureTransformation.ONEHOT:
            return OneHot(feature_props['vocab_size'])
        elif transformation_type == FeatureTransformation.BINNING:
            return HistogramBinning(feature_props['boundaries'])
        elif transformation_type == FeatureTransformation.STANDARDIZATION:
            return Standardization(feature_props['mean'], feature_props['stddev'])
        else:
            raise Exception(f'Wrong transformation type - {transformation_type}')


    def _parse(self, conf: Dict[str, object]) -> Dict[str, BaseTransformation]:
        if 'features' not in conf:
            raise Exception('Please define "features".')

        transformation_by_feature = {}
        if 'transformation' in conf:
            for feature, props in conf['transformation'].items():
                transformation_type = FeatureTransformation(props['type'])
                required_fields = self._get_required_fields_by_type(transformation_type)

                feature_props = props['properties']
                if required_fields not in feature_props:
                    raise Exception(f'Required fields for {feature}: {required_fields}')

                transformation = self._parse_transformation(transformation_type, feature_props)
                transformation_by_feature[feature] = transformation

        return transformation_by_feature
