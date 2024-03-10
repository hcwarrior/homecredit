import tensorflow.keras as tf_keras

from base_transformation import BaseTransformation


class Standardization(BaseTransformation):
    def __init__(self, mean: float, stddev: float):
        self.layer = tf_keras.layers.Normalization(mean=mean, variance=stddev)

    def get_layer(self):
        return self.layer
