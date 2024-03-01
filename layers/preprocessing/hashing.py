import tensorflow as tf


class Hashing(tf.keras.layers.Layer):
    def __init__(self, num_bins):
        super().__init__()

        # uses Siphash64
        self.hashing_layer = tf.keras.layers.Hashing(
            num_bins=num_bins, salt=42, output_mode='int')

    def call(self, inputs):
        return self.hashing_layer(inputs)
