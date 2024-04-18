from typing import Dict, Optional

import tensorflow as tf
import tensorflow.keras as tf_keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.python.framework.dtypes import DType

class DeepCrossNetwork(tf_keras.Model):
    def __init__(self,
                 transformations_by_feature: Dict[str, object],
                 **kwargs):
        super(DeepCrossNetwork, self).__init__(**kwargs)
        self.transformations_by_feature = transformations_by_feature
        self._build_layers()


    def _parse_into_layer(self, transformation: Dict[str, object]):
        type = transformation['type']
        feature_props = transformation['properties']

        if type == 'numerical_embedding':
            return tf_keras.layers.Embedding(feature_props['vocab_size'], feature_props['embedding_size'])
        elif type == 'onehot':
            return tf_keras.layers.StringLookup(vocabulary=feature_props['vocab'] + ['NA'], output_mode='one_hot')
        elif type == 'binning':
            return tf_keras.layers.Discretization(bin_boundaries=feature_props['boundaries'])
        elif type == 'standardization':
            tf_keras.layers.Normalization(mean=feature_props['mean'], variance=feature_props['stddev'])
        else:
            return tf_keras.layers.Identity()

    def _build_layers(self):
        input_by_feature_name, transformed_by_feature_name = {}, {}
        for feature, transformation in self.transformations_by_feature.items():
            transformation = self._parse_into_layer(transformation)
            dtype = self._get_dtype_by_transformation(transformation)
            inputs_placeholder = tf_keras.Input((1, ),
                                                dtype=dtype,
                                                name=feature)
            transformed = tf.cast(transformation(inputs_placeholder), tf.float32)

            input_by_feature_name[feature] = inputs_placeholder
            transformed_by_feature_name[feature] = transformed

        concatenated_input = tf_keras.layers.Concatenate(axis=-1)(list(transformed_by_feature_name.values()))
        cross_layer_output = self._build_cross_layers(concatenated_input)
        logits = self._build_dense_layers(inputs=cross_layer_output)

        self.model = Model(input_by_feature_name, logits)

    def _get_dtype_by_transformation(self, transformation: tf_keras.layers.Layer) -> DType:
        if isinstance(transformation, tf_keras.layers.StringLookup):
            return tf.string
        return tf.float32

    def _build_cross_layers(self, x0):
        # TODO: Please add parameters
        x1 = self._cross(x0, x0)
        x2 = self._cross(x0, x1)

        return x2


    def _cross(self,
               x0,
               x1,
               use_bias: bool = True,
               activation: tf_keras.layers.Activation = None,
               kernel_initializer: tf_keras.initializers.Initializer = tf_keras.initializers.truncated_normal,
               bias_initializer: tf_keras.initializers.Initializer = tf_keras.initializers.zeros,
               kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
               bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None) -> tf.Tensor:
        layer = tf_keras.layers.Dense(
            x0.shape[-1],
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            use_bias=use_bias,
            dtype=x0.dtype,
            activation=activation,
        )

        result = layer(x1)
        result = tf.cast(result, x0.dtype)

        return x0 * result + result


    def _build_dense_layers(self, inputs):
        # TODO: Please add parameters
        layer1 = Dense(50, activation=tf_keras.activations.relu)
        layer2 = Dense(30, activation=tf_keras.activations.relu)

        output = layer2(layer1(inputs))
        logits = Dense(units=1, activation=tf_keras.activations.sigmoid)(output)

        return logits

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        return {"transformations_by_feature": self.transformations_by_feature}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

