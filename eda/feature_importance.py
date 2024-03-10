import xgboost
from pandas import DataFrame
from xgboost import XGBClassifier, XGBModel


class TreeBasedFeatureSelector:
    # assume memory is enough
    def fit(self, X: DataFrame, y: DataFrame):
        classifier = XGBClassifier(booster='gbtree', importance_type='gain')
        classifier.fit(X, y)
        print(classifier.feature_importances())
        return classifier

    def plot(self, xgb: XGBModel):
        xgboost.plot_importance(xgb, importance_type='gain', title='gain', xlabel='', grid=False)
