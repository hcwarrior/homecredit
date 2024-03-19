from abc import ABCMeta, abstractmethod

import tensorflow.keras as tf_keras


class BaseTransformation(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, num_hashing_bins: int):
        if num_hashing_bins is not None and num_hashing_bins > 0:
            self.hashing_layer = tf_keras.layers.Hashing(num_bins=num_hashing_bins)

    @abstractmethod
    def forward(self, input_tensor: tf_keras.Input):
        pass
