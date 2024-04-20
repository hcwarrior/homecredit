from typing import Dict
from dataset.feature.feature import *
from dataset.feature.feature_definer import FeatureDefiner
from dataset.datainfo import DATA_PATH
from dataset import const

period_col: Dict[str, List[str]] = {
    'applprev': ['creationdate_885D'],
    'credit_bureau_a': ['dateofcredstart_181D', 'dateofcredstart_739D'],
    'credit_bureau_b': ['contractdate_551D'],
    'person': None,
    'debitcard': ['openingdate_857D'],
    'deposit': ['openingdate_313D'],
    'other': None,
    'tax_registry_a': ['recorddate_4527225D'],
    'tax_registry_b': ['deductiondate_4917603D'],
    'tax_registry_c': ['processingdate_168D'],
}

if __name__ == '__main__':
    # create folder for feature definition
    os.makedirs(DATA_PATH / 'feature_definition', exist_ok=True)

    # define features for each topic
    for topic in const.TOPICS:
        if topic.depth == 1:
            print(f'[*] Defining features for {topic.name}')
            fd = FeatureDefiner(topic.name, period_cols=period_col.get(topic.name, None))
            features = fd.define_features()
            print(f'{topic.name} has {len(features)} features')
            FeatureDefiner.save_json(features, DATA_PATH / f'feature_definition/{topic.name}.json')
