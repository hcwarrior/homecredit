# Homecredit repository
## Pre-requirements
The following binaries should be installed.
```
python==3.9
Anaconda (or Miniconda)
```
Please install libraries using the following command.
```shell
# conda-forge offers more libraries
conda config --add channels conda-forge
conda create -n homecredit
conda activate homecredit
conda install --yes -c conda-forge --file requirements.txt
```

## Run
```shell
python3 run.py --feature_yaml_path './examples/features.yml' --model_yaml_path './examples/model.yml' \
  --train_data_root_dir '~/data/train' --val_data_root_dir '~/data/val' --test_data_root_dir '~/data/val' \
  --submission_csv_file_path '~/submission.csv'   
```
