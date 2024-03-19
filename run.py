from dataclasses import dataclass
from simple_parsing import ArgumentParser

from parsing.feature.feature_parser import FeatureParser


@dataclass
class Options:
    feature_yaml_path: str # A feature YAML file path


def _parse_features(feature_yaml_path: str):
    feature_parser = FeatureParser()
    feature_parser.load_prop(feature_yaml_path)

    print(feature_parser.conf)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")

    args = parser.parse_args()
    options = args.options

    _parse_features(options.feature_yaml_path)
