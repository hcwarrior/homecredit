import tensorflow.keras as tf_keras

from base_transformation import BaseTransformation


class Embedding(BaseTransformation):
    def __init__(self, vocab_size: int, embedding_size: int):
        self.layer = tf_keras.layers.Embedding(vocab_size, embedding_size)

    def get_layer(self):
        return self.layer
