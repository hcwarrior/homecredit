from typing import Dict, Set

import tensorflow.keras as tf_keras
import yaml

from parsing.feature.transformation_type import FeatureTransformation


class FeatureParser:
    def __init__(self):
        self.conf = {}

    def load_prop(self, features_file_path):
        with open(features_file_path) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            self.conf = self._parse(conf)

    def _get_required_fields_by_type(self, transformation_type: FeatureTransformation) -> Set[str]:
        if transformation_type in {FeatureTransformation.NUMERICAL_EMBEDDING,
                                   FeatureTransformation.CHARACTER_EMBEDDING}:
            return {'vocab_size', 'embedding_size'}
        elif transformation_type == FeatureTransformation.ONEHOT:
            return {'vocab'}
        elif transformation_type == FeatureTransformation.BINNING:
            return {'boundaries'}
        elif transformation_type == FeatureTransformation.STANDARDIZATION:
            return {'mean', 'stddev'}
        elif transformation_type == FeatureTransformation.TARGET_ENCODING:
            return {'value', 'encoded'}
        elif transformation_type == FeatureTransformation.RAW:
            return set()
        else:
            raise Exception(f'Wrong transformation type - {transformation_type}')


    def _parse(self, conf: Dict[str, object]) -> Dict[str, object]:
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

                transformation_by_feature[feature] = props

        return transformation_by_feature
