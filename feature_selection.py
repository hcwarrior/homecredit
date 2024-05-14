from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import yaml

from matplotlib import pyplot as plt
from simple_parsing import ArgumentParser
from sklearn.ensemble import RandomForestClassifier


@dataclass
class SelectionOptions:
    feature_conf_yaml_path: str
    data_parquet_root_dir: str
    feature_imps_png_output_path: str
    top_k: int
    down_sampling_ratio: float = 1.0


# returns continuous, categorical features and label in order
def _parse_feature_conf(feature_conf_yaml_path: str) -> Tuple[List[str], List[str], str]:
    with open(feature_conf_yaml_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        features = conf['features']
        return features['continuous'], features['categorical'], features['label']


def _get_preprocessed_dataframe(
        feature_conf_yaml_path: str,
        data_parquet_root_dir: str,
        down_sampling_ratio: float):
    MIN_CATEGORIES_FOR_TARGET_ENCODING = 1
    cont_features, cat_features, label = _parse_feature_conf(feature_conf_yaml_path)
    features = cont_features + cat_features + [label]
    df = pd.read_parquet(data_parquet_root_dir, columns=features, engine='pyarrow')
    df = df.sample(frac=down_sampling_ratio, random_state=42)

    # fill NA values
    df.bfill(inplace=True)

    # simple standardization
    for cont_feature in cont_features:
        df[cont_feature] = (df[cont_feature] - df[cont_feature].mean()) / df[cont_feature].std()

    return df, cont_features, cat_features, label


def _drop_correlated_features(
        df: pd.DataFrame,
        cat_features: List[str],
        threshold: float = 0.9) -> pd.DataFrame:
    df_new = df[cont_features]
    for cat_feature in cat_features:
        target_means = df[[cat_feature, label]].groupby(cat_feature, observed=False)[label].mean().fillna(0)
        df_new.loc[:, cat_feature] = df[cat_feature].map(target_means)

    corr_mat = np.corrcoef(df_new.values, rowvar=False)
    print('Correlation')
    print(corr_mat)
    columns = df_new.columns
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
    print(X.head())
    X[cat_features] = X[cat_features].astype('category')
    clf = xgb.XGBClassifier(tree_method="hist", enable_categorical=True)
    clf.fit(X, y)

    print('Fitted Random Forest.')
    imps = pd.Series(clf.feature_importances_, index=X.columns)
    imps = imps.sort_values(ascending=False)[:top_k]

    print('Feature Importances (Sorted)')
    print(imps)

    fig, ax = plt.subplots()
    imps.plot.bar(ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.gcf().set_size_inches(50, 20)
    plt.savefig(feature_imps_png_output_path, dpi=199)

    cont_feature_set = set(cont_features)
    selected_cont_features, selected_cat_features = [], []
    for imp in imps.items():
        col = imp[0]
        if col in cont_feature_set:
            selected_cont_features.append(col)
        else:
            selected_cat_features.append(col)

    return selected_cont_features, selected_cat_features


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(SelectionOptions, dest="options")

    args = parser.parse_args()
    options = args.options

    df, cont_features, cat_features, label = _get_preprocessed_dataframe(
        options.feature_conf_yaml_path, options.data_parquet_root_dir, options.down_sampling_ratio)

    print('Preprocessed DataFrame')
    print(df.head(5))

    df = _drop_correlated_features(df, cat_features, 0.8)

    print(f'Columns after dropping highly correlated features: {df.columns}')

    cont_features = list(set(cont_features) & set(df.columns))
    cat_features = list(set(cat_features) & set(df.columns))
    cont_features, cat_features = _select_features(df, label, cont_features, cat_features, options.feature_imps_png_output_path, options.top_k)

    print(f'Selected Continuous Features: [{", ".join(cont_features)}]')
    print(f'Selected Categorical Features: [{", ".join(cat_features)}]')

    # Optional - class weights
    weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=df[label].values)
    weights = dict(zip([0, 1], weights))
    print(f'Weights: {weights}')
