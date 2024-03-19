from typing import Dict

import tensorflow as tf
import tensorflow.keras as tf_keras
from keras import Model
from keras.layers import Dense

from layers.model.cross import Cross
from layers.transformation.base_transformation import BaseTransformation


class DeepCrossNetwork(tf_keras.Model):
    def __init__(self,
                 transformations_by_feature: Dict[str, BaseTransformation]):
        super().__init__()
        self.transformations_by_feature = transformations_by_feature
        self._build_layers()


    def _build_layers(self):
        features = []
        for feature, transformation in self.transformations_by_feature.items():
            inputs_placeholder = tf_keras.Input((None, ),
                           name=feature,
                           dtype=tf.float32)

            transformed = transformation(inputs_placeholder)
            features.append(transformed)
        x0 = tf.concat(features, axis=-1)
        x1 = Cross(x0, x0)
        x2 = Cross(x0, x1)
        logits = Dense(units=2)(x2)
        self.model = Model(features, logits)


    def call(self, inputs: tf_keras.Input):
        x = self.dense1(inputs)
        return self.dense2(x)
