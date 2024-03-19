import tensorflow.keras as tf_keras

from layers.transformation.base_transformation import BaseTransformation


class Standardization(BaseTransformation):
    def __init__(self, mean: float, stddev: float, num_hashing_bins: int):
        super().__init__(num_hashing_bins)
        self.layer = tf_keras.layers.Normalization(mean=mean, variance=stddev)

    def call(self, inputs: tf_keras.Input):
        if self.hashing_layer is not None:
            inputs = self.hashing_layer(inputs)
        return self.layer(inputs)
