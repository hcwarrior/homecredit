import tensorflow.keras as tf_keras

from layers.transformation.base_transformation import BaseTransformation


class Raw(BaseTransformation):
    def __init__(self, num_hashing_bins: int):
        super().__init__(num_hashing_bins)

    def forward(self, input_tensor: tf_keras.Input):
        if self.hashing_layer is not None:
            input_tensor = self.hashing_layer(input_tensor)
        return input_tensor
