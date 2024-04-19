import tensorflow as tf
import tensorflow.keras as tf_keras

from layers.transformation.base_transformation import BaseTransformation


class CharacterEmbedding(BaseTransformation):
    def __init__(self, vocab_size: int, embedding_size: int, num_hashing_bins: int):
        super().__init__(num_hashing_bins)
        # this is added to handle null values (NA)
        vocab_size = vocab_size + 1
        self.labeling_layer = tf_keras.layers.Hashing(vocab_size)
        self.layer = tf_keras.layers.Embedding(vocab_size, embedding_size)

    def call(self, inputs: tf_keras.Input):
        if self.hashing_layer is not None:
            inputs = self.hashing_layer(inputs)
        inputs = self.labeling_layer(tf.cast(inputs, tf.string))
        return tf.squeeze(self.layer(inputs), axis=1)