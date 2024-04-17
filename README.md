# Homecredit repository
## Pre-requirements
The following binaries should be installed.
```
python==3.9
pip3
```
Please install libraries using the following command.
```shell
pip3 install -r requirements.txt
```

## Run
```shell
python3 run.py --feature_yaml_path './examples/features.yml' --model_yaml_path './examples/model.yml' \
  --train_data_root_dir '~/data/train' --val_data_root_dir '~/data/val' --test_data_root_dir '~/data/val' \
  --submission_csv_file_path '~/submission.csv'   
```
