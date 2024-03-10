from typing import List

import tensorflow.keras as tf_keras

from base_transformation import BaseTransformation


class HistogramBinning(BaseTransformation):
    def __init__(self, bin_boundaries: List[float]):
        self.layer = tf_keras.layers.Discretization(bin_boundaries=bin_boundaries)

    def get_layer(self):
        return self.layer
