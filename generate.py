from dataclasses import dataclass

from simple_parsing import ArgumentParser

from generator.feature_yaml_generator_tree import FeatureYAMLGeneratorTree


@dataclass
class GenerateOptions:
    feature_conf_yaml_path: str
    data_parquet_file_path: str
    output_yaml_path: str

def _generate_feature_yaml(
        feature_conf_yaml_path: str,
        data_parquet_file_path: str,
        output_yaml_path: str):
    feature_yaml_generator = FeatureYAMLGeneratorTree(
        feature_conf_yaml_path, data_parquet_file_path, output_yaml_path)
    feature_yaml_generator.generate()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(GenerateOptions, dest="options")

    args = parser.parse_args()
    options = args.options

    _generate_feature_yaml(
        options.feature_conf_yaml_path,
        options.data_parquet_file_path,
        options.output_yaml_path)
