import tensorflow.keras as tf_keras

from layers.transformation.base_transformation import BaseTransformation


class Raw(BaseTransformation):
    def __init__(self, num_hashing_bins: int):
        super().__init__(num_hashing_bins)

    def forward(self, inputs: tf_keras.Input):
        if self.hashing_layer is not None:
            inputs = self.hashing_layer(inputs)
        return inputs
