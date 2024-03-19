from typing import Optional

from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow.keras as tf_keras


class Cross(tf_keras.layers.Layer):
  def __init__(
      self,
      use_bias: bool = True,
      activation: layers.Activation = None,
      kernel_initializer: initializers.Initializer = initializers.truncated_normal,
      bias_initializer: initializers.Initializer = initializers.zeros,
      kernel_regularizer: Optional[regularizers.Regularizer] = None,
      bias_regularizer: Optional[regularizers.Regularizer] = None,
      **kwargs):
    super(Cross, self).__init__(**kwargs)

    self._use_bias = use_bias
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._input_dim = None

    self._supports_masking = True

  def build(self, input_shape):
    self._layer = layers.Dense(
        input_shape[-1],
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        use_bias=self._use_bias,
        dtype=self.dtype,
        activation=self._activation,
    )
    self.built = True

  def call(self, x0: tf.Tensor, x1: tf.Tensor) -> tf.Tensor:
    assert x0.shape == x1.shape

    if not self.built:
      self.build(x0.shape)

    result = self._layer(x1)
    result = tf.cast(result, self.compute_dtype)

    return x0 * result + result
