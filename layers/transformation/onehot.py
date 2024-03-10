import tensorflow.keras as tf_keras

from base_transformation import BaseTransformation


class OneHot(BaseTransformation):
    def __init__(self, vocab_size: int):
        self.layer = tf_keras.layers.CategoryEncoding(num_tokens=vocab_size, output_mode='one_hot')

    def get_layer(self):
        return self.layer
