from abc import abstractmethod
from typing import Optional

import tensorflow.keras as tf_keras


class BaseTransformation(tf_keras.layers.Layer):
    def __init__(self, num_hashing_bins: int):
        super(BaseTransformation, self).__init__()
        self.hashing_layer = None
        if num_hashing_bins is not None and num_hashing_bins > 0:
            self.hashing_layer = tf_keras.layers.Hashing(num_bins=num_hashing_bins)
        self.layer: Optional[tf_keras.layers.Layer] = None

    @abstractmethod
    def call(self, inputs: tf_keras.Input):
        pass
