from typing import List

import tensorflow as tf
import tensorflow.keras as tf_keras

from layers.transformation.base_transformation import BaseTransformation


class OneHot(BaseTransformation):
    def __init__(self, vocab: List[str], num_hashing_bins: int):
        super().__init__(num_hashing_bins)
        self.layer = tf_keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')

    def call(self, inputs: tf_keras.Input):
        if self.hashing_layer is not None:
            inputs = self.hashing_layer(inputs)
        return self.layer(tf.cast(inputs, tf.string))
