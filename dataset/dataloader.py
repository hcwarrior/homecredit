import json
from pyspark.sql import SparkSession, DataFrame


class SparkDataLoader:
    def __init__(self, spark_session :SparkSession, data_path: str = "/home/w1/work/data/parquet_files/train"):
        self.spark = spark_session
        self.data_path = data_path
        self.temp_table_name = "tmp"

    def get_field_agg_map(self, df: DataFrame):
        field_schemas = json.loads(df.schema.json())["fields"]
        field_type_map = {field["name"]: field["type"] for field in field_schemas}

        cast_map = dict()
        for field, field_type in field_type_map.items():
            if field == "case_id":
                continue

            if field_type == "string":
                cast_map[field] = f"max({field}) as {field}"
            elif field_type in ("integer", "long", "short", "double"):
                cast_map[field] = f"avg({field}) as {field}"
            elif field_type == "boolean":
                cast_map[field] = f"avg(cast({field} as int)) as {field}"
            else:
                cast_map[field] = f"max({field}) as {field}"
        return cast_map
    
    def load_with_merge_schema(self) -> DataFrame:
        return self.spark.read.parquet(self.data_path, mergeSchema=True)

    def get_agg_data(self, df :DataFrame) -> DataFrame:
        df.createOrReplaceTempView(self.temp_table_name)
        field_agg_map = self.get_field_agg_map(df)

        group_by_sql = f"""
                        SELECT case_id,
                        {','.join(field_agg_map.values())}
                        FROM {self.temp_table_name}
                        GROUP BY case_id    
                        """

        df_agg = self.spark.sql(group_by_sql)
        return df_agg
    
    def write_agg_date(self, df: DataFrame, path: str):
        df_agg = self.get_agg_data(df)
        df_agg.write.parquet(path, compression="gzip", mode="overwrite")


spark = SparkSession.builder.appName("example").config("spark.driver.memory", "20g").getOrCreate()
loader = SparkDataLoader(spark, "/home/w1/work/data/parquet_files/train")

df = loader.load_with_merge_schema()
df.sort("case_id").write.parquet("/tmp/t1",compression="gzip", mode="overwrite")

df = spark.read.parquet("/tmp/t1")
loader.get_agg_data(df).write.parquet("/tmp/t2",compression="gzip",mode="overwrite")


df=spark.read.parquet("/tmp/t2").toPandas()