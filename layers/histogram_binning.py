import tensorflow as tf


class HistogramBinning(tf.keras.layers.Layer):
    def __init__(self, num_bins):
        super().__init__()
        self.binning_layer = tf.keras.layers.Discretization(
            num_bins=num_bins, epsilon=0.01, output_mode='int')

    def call(self, inputs):
        return self.binning_layer(inputs)
