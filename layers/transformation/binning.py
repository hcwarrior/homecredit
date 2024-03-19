from typing import List

import tensorflow.keras as tf_keras

from layers.transformation.base_transformation import BaseTransformation


class HistogramBinning(BaseTransformation):
    def __init__(self, bin_boundaries: List[float], num_hashing_bins: int):
        super().__init__(num_hashing_bins)
        self.layer = tf_keras.layers.Discretization(bin_boundaries=bin_boundaries)

    def forward(self, input_tensor: tf_keras.Input):
        if self.hashing_layer is not None:
            input_tensor = self.hashing_layer(input_tensor)
        return self.layer(input_tensor)
