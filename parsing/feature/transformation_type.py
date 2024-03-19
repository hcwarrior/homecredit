from enum import Enum


class FeatureTransformation(Enum):
    EMBEDDING = 'embedding'
    ONEHOT = 'onehot'
    BINNING = 'binning'
    STANDARDIZATION = 'standardization'

