# requirements for dataset module

## define features
define all logical possible features
definer has
 - naming rule
 - predefined processing type

input : 
 - datasource
 - processing type
output :
 - Dict[feature_name : definition]


## build features
build feature specific logic
 - predefined processing logic

input : 
 - feture_name
 - definition
output :
 - Dict[feature_name : query or sth]


## transformation
from raw dataset to 1d dataset

input : parquet_file, feature_list
output : store parquet file or directly return pl.DataFrame

### transformation for selected feature
for efficiency of final model training and inference
dependency of pre-processing

