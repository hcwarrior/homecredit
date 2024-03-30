import tensorflow as tf
import tensorflow.keras as tf_keras

from layers.transformation.base_transformation import BaseTransformation


class Raw(BaseTransformation):
    def __init__(self, num_hashing_bins: int):
        super().__init__(num_hashing_bins)

    def call(self, inputs: tf_keras.Input):
        if self.hashing_layer is not None:
            inputs = self.hashing_layer(inputs)

        if len(inputs.shape) < 2:
            inputs = tf.reshape(inputs, (-1, 1))
        return inputs
