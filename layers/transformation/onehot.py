import tensorflow.keras as tf_keras

from layers.transformation.base_transformation import BaseTransformation


class OneHot(BaseTransformation):
    def __init__(self, vocab_size: int, num_hashing_bins: int):
        super(num_hashing_bins)
        self.layer = tf_keras.layers.CategoryEncoding(num_tokens=vocab_size, output_mode='one_hot')

    def forward(self, input_tensor: tf_keras.Input):
        if self.hashing_layer is not None:
            input_tensor = self.hashing_layer(input_tensor)
        return self.layer(input_tensor)
