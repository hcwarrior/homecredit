from enum import Enum


class FeatureTransformation(Enum):
    CHARACTER_EMBEDDING = 'character_embedding'
    NUMERICAL_EMBEDDING = 'numerical_embedding'
    ONEHOT = 'onehot'
    BINNING = 'binning'
    STANDARDIZATION = 'standardization'
    RAW = 'raw'

