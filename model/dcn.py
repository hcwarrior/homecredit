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

        concatenated_input = tf_keras.layers.Concatenate(axis=-1)(list(input_by_feature_name.values()))
        cross_layer_output = self._build_cross_layers(concatenated_input)
        logits = self._build_dense_layers(inputs=cross_layer_output)

        self.model = Model(input_by_feature_name, logits)

    def _build_cross_layers(self, x0):
        # TODO: Please add parameters
        x1 = Cross()(x0, x0)
        x2 = Cross()(x0, x1)

        return x2

    def _build_dense_layers(self, inputs):
        # TODO: Please add parameters
        layer1 = Dense(50, activation=tf_keras.activations.relu)
        layer2 = Dense(30, activation=tf_keras.activations.relu)

        output = layer2(layer1(inputs))
        logits = Dense(units=1, activation=tf_keras.activations.sigmoid)(output)

        return logits


    def call(self, inputs):
        return self.model(inputs)
