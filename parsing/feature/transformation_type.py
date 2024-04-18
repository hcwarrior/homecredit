from enum import Enum


class FeatureTransformation(Enum):
    NUMERICAL_EMBEDDING = 'numerical_embedding'
    ONEHOT = 'onehot'
    BINNING = 'binning'
    STANDARDIZATION = 'standardization'
    RAW = 'raw'

