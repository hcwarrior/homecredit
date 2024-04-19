from typing import Dict
from dataset.feature.feature import *
from dataset.feature.feature_definer import FeatureDefiner
from dataset import const

period_col: Dict[str, List[str]] = {
    'applprev': ['creationdate_885D'],
    # 'credit_bureau_a': ['creationdate_885D'],
    # 'credit_bureau_b': ['creationdate_885D'],
    # 'person': ['creationdate_885D'],
    # 'debitcard': ['creationdate_885D'],
    # 'deposit': ['creationdate_885D'],
    # 'other': ['creationdate_885D'],
    # 'tax_registry_a': ['creationdate_885D'],
    # 'tax_registry_b': ['creationdate_885D'],
    # 'tax_registry_c': ['creationdate_885D'],
}

if __name__ == '__main__':
    # create folder for feature definition
    os.makedirs('data/feature_definition', exist_ok=True)

    # define features for each topic
    for topic in const.TOPICS:
        if topic.depth == 1 and topic.name in ('debitcard', 'deposit', 'other', 'tax_registry_a', 'tax_registry_b', 'tax_registry_c'):
            fd = FeatureDefiner(topic.name, period_cols=period_col.get(topic.name, None))
            features, _, _ = fd.define_features()
            print(f'{topic.name} has {len(features)} features')
            FeatureDefiner.save_json(features, f'data/feature_definition/{topic.name}.json')

