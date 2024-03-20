from typing import Dict

import tensorflow as tf
import tensorflow.keras as tf_keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

from layers.model.cross import Cross
from layers.transformation.base_transformation import BaseTransformation


class DeepCrossNetwork(tf_keras.Model):
    def __init__(self,
                 transformations_by_feature: Dict[str, BaseTransformation]):
        super().__init__()
        self.transformations_by_feature = transformations_by_feature
        self._build_layers()


    def _build_layers(self):
        input_by_feature_name, transformed_by_feature_name = {}, {}
        for feature, transformation in self.transformations_by_feature.items():
            inputs_placeholder = tf_keras.Input((1, ), name=feature)
            transformed = tf.cast(transformation(inputs_placeholder), tf.float32)

            input_by_feature_name[feature] = inputs_placeholder
            transformed_by_feature_name[feature] = transformed

        # TODO: Please add parameters
        x0 = tf_keras.layers.Concatenate(axis=-1)(list(transformed_by_feature_name.values()))
        x1 = Cross()(x0, x0)
        x2 = Cross()(x0, x1)
        logits = Dense(units=2)(x2)
        self.model = Model(input_by_feature_name, logits)


    def call(self, inputs):
        return self.model(inputs)
