import pickle
from optbinning import OptimalBinning
from sklearn.preprocessing import LabelEncoder
import numpy as np

class BinNormEncoder:
    def __init__(self, criteria_path) -> None:
        self.criteria_path = criteria_path

    def fit(self, X, y):
        criteria = {}
        for column in X.columns:
            optb = OptimalBinning(name=column, dtype="numerical", solver="cp")
            optb.fit(X[column], y)
            binned_data = optb.transform(X[column].values, metric="bins")

            le = LabelEncoder()
            encoded_data = le.fit_transform(binned_data)
            max_encoded_value = np.max(encoded_data)

            criteria[column] = {
                "optb": optb,
                "le": le,
                "max_encoded": max_encoded_value
            }
        
        with open(self.criteria_path, "wb") as f:
            pickle.dump(criteria, f)
    
    def transform(self, X):
        with open(self.criteria_path, "rb") as f:
            criteria = pickle.load(f)
        
        for column in X.columns:
            optb = criteria[column]["optb"]
            le = criteria[column]["le"]
            max_encoded_value = criteria[column]["max_encoded"]

            binned_data = optb.transform(X[column].values, metric="bins")
            encoded_data = le.transform(binned_data)
            normalized_data = encoded_data / (max_encoded_value + 1e-10)

            X[column] = encoded_data