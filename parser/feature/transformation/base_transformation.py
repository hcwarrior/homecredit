from abc import ABCMeta, abstractmethod

import tensorflow.keras as tf_keras


class BaseTransformation(metaclass=ABCMeta):

    @abstractmethod
    def get_layer(self):
        pass

    def transform(self, input_tensor: tf_keras.Input):
        return self.get_layer()(input_tensor)
