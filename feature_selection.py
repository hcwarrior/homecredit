from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
import yaml

from matplotlib import pyplot as plt
from simple_parsing import ArgumentParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class SelectionOptions:
    feature_conf_yaml_path: str
    data_parquet_file_path: str
    feature_imps_png_output_path: str
    top_k: int


# returns continuous, categorical features and label in order
def _parse_feature_conf(feature_conf_yaml_path: str) -> Tuple[List[str], List[str], str]:
    with open(feature_conf_yaml_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        features = conf['features']
        return features['continuous'], features['categorical'], features['label']


def _get_preprocessed_dataframe(
        feature_conf_yaml_path: str,
        data_parquet_file_path: str):
    cont_features, cat_features, label = _parse_feature_conf(feature_conf_yaml_path)
    features = cont_features + cat_features + [label]
    df = pd.read_parquet(data_parquet_file_path, columns=features, engine='fastparquet')
    df.bfill(inplace=True)

    # Use label encoding for categorical features
    le = LabelEncoder()
    df[cat_features] = df[cat_features].apply(le.fit_transform)

    return df, cont_features, cat_features, label


def _drop_correlated_features(
        df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    corr_mat = np.corrcoef(df.values, rowvar=False)
    print('Correlation')
    print(corr_mat)
    columns = df.columns

    drop_columns = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if corr_mat[i][j] >= threshold:
                drop_columns.append(columns[j])

    return df.drop(columns=drop_columns)


def _select_features(
        df: pd.DataFrame,
        label: str,
        cont_features: List[str],
        cat_features: List[str],
        feature_imps_png_output_path: str,
        top_k: int) -> Tuple[List[str], List[str]]:
    X, y = df.drop(columns=[label]), df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    features = df.columns
    cont_feature_set, cat_feature_set = set(cont_features), set(cat_features)
    feature_imps = {features[i]: np.mean(np.abs(shap_values[:, i])) for i in range(len(features))}
    feature_imps_ordered = OrderedDict(sorted(feature_imps.items(), key=lambda x: x[1]))

    feature_imps = feature_imps_ordered.items()[:top_k]

    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(feature_imps_png_output_path)

    selected_cont_features, selected_cat_features = [], []
    for imp in feature_imps:
        if imp in cont_feature_set:
            selected_cont_features.append(imp[0])
        else:
            selected_cat_features.append(imp[0])

    return selected_cont_features, selected_cat_features


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(SelectionOptions, dest="options")

    args = parser.parse_args()
    options = args.options

    df, cont_features, cat_features, label = _get_preprocessed_dataframe(
        options.feature_conf_yaml_path, options.data_parquet_file_path)

    print('Preprocessed DataFrame')
    print(df.head(5))

    df = _drop_correlated_features(df, 0.8)

    print(f'Columns after dropping highly correlated features: {df.columns.tolist()}')

    cont_features, cat_features = _select_features(df, label, cont_features, cat_features, options.feature_imps_png_output_path, options.top_k)

    print(f'Selected Continuous Features: [{", ".join(cont_features)}]')
    print(f'Selected Categorical Features: [{", ".join(cat_features)}]')
