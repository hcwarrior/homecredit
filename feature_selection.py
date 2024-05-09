from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
import sklearn
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

    # Use label encoding for categorical features
    onehot_cat_features, target_encoding_cat_features = [], []
    for cat_feature in cat_features:
        # target encoding (mean)
        if df[cat_feature].nunique() >= MIN_CATEGORIES_FOR_TARGET_ENCODING:
            target_means = df[[cat_feature, label]].groupby(cat_feature, observed=False)[label].mean()
            df[cat_feature] = df[cat_feature].map(target_means)
            target_encoding_cat_features.append(cat_feature)
        else:
            dummy_df = pd.get_dummies(df[cat_feature], prefix=cat_feature, dtype=float)
            df = pd.concat([df, dummy_df], axis=1)
            df.drop(columns=[cat_feature], inplace=True)
            onehot_cat_features = onehot_cat_features + list(dummy_df.columns)

    # simple standardization
    for cont_feature in cont_features:
        df[cont_feature] = (df[cont_feature] - df[cont_feature].mean()) / df[cont_feature].std()

    return df, cont_features, onehot_cat_features, target_encoding_cat_features, label


def _drop_correlated_features(
        df: pd.DataFrame,
        onehot_cat_features: Set[str],
        threshold: float = 0.8) -> pd.DataFrame:
    corr_mat = np.corrcoef(df.values, rowvar=False)
    print('Correlation')
    print(corr_mat)
    columns = df.columns
    idxs_by_onehot_col = defaultdict(list)
    for idx, col in enumerate(columns):
        if col in onehot_cat_features:
            original_col = col.rsplit('_', 1)[0]
            idxs_by_onehot_col[original_col].append(idx)

    drop_columns = []
    skip_idxs = set()
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if columns[j] in onehot_cat_features and j not in skip_idxs:
                original_col = columns[j].rsplit('_', 1)[0]
                onehot_col_idxs = idxs_by_onehot_col[original_col]
                corr = corr_mat[i][onehot_col_idxs].max()
                if corr >= threshold:
                    skip_idxs.update(idxs_by_onehot_col[original_col])
                    drop_columns = drop_columns + list(columns.to_series().iloc[onehot_col_idxs])
                    continue

            if corr_mat[i][j] >= threshold:
                drop_columns.append(columns[j])

    return df.drop(columns=drop_columns)


def _select_features(
        df: pd.DataFrame,
        label: str,
        cont_features: List[str],
        onehot_cat_features: Set[str],
        target_encoding_cat_features: Set[str],
        feature_imps_png_output_path: str,
        top_k: int) -> Tuple[List[str], List[str]]:
    X, y = df.drop(columns=[label]), df[label]
    print(X.head())
    rf = RandomForestClassifier(n_estimators=50, max_depth=3, class_weight='balanced', random_state=42)
    rf.fit(X, y)

    print('Fitted Random Forest.')
    imps_aggregated = defaultdict(float)
    imps = pd.Series(rf.feature_importances_, index=X.columns)
    for col in imps.index:
        if col in onehot_cat_features:
            original_col = col.rsplit('_', 1)[0]
            imps_aggregated[original_col] += imps[col]
            print(col, imps_aggregated[original_col])
        else:
            imps_aggregated[col] = imps[col]

    imps_aggregated = pd.Series(imps_aggregated.values(), index=imps_aggregated.keys())
    imps = imps_aggregated.sort_values(ascending=False)[:top_k]

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

    df, cont_features, onehot_cat_features, target_encoding_cat_features, label = _get_preprocessed_dataframe(
        options.feature_conf_yaml_path, options.data_parquet_root_dir, options.down_sampling_ratio)

    print('Preprocessed DataFrame')
    print(df.head(5))

    df = _drop_correlated_features(df, set(onehot_cat_features), 0.8)

    print(f'Columns after dropping highly correlated features: {df.columns}')

    cont_features, cat_features = _select_features(df, label, cont_features, set(onehot_cat_features), set(target_encoding_cat_features), options.feature_imps_png_output_path, options.top_k)

    print(f'Selected Continuous Features: [{", ".join(cont_features)}]')
    print(f'Selected Categorical Features: [{", ".join(cat_features)}]')

    # Optional - class weights
    weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=df[label].values)
    weights = dict(zip([0, 1], weights))
    print(f'Weights: {weights}')
