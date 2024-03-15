import os
import glob
import polars as pl


class ParquetDataLoader:
    def __init__(self, data_path: str = "/home/w1/work/data/parquet_files/train"):
        self.data_path = data_path
        self.files =  [os.path.abspath(file) for file in glob.glob(f"{self.data_path}/*.parquet")]

    def get_field_agg_map(self, lf: pl.LazyFrame):
        cast_map = dict()
        for field, field_type in lf.schema.items():
            if field == "case_id":
                continue

            if field_type == pl.String:
                cast_map[field] = f"max({field}) as {field}"
            elif field_type.is_numeric():
                cast_map[field] = f"avg({field}) as {field}"
            elif field_type == pl.Boolean:
                cast_map[field] = f"avg(cast({field} as int)) as {field}"
            else:
                cast_map[field] = f"max({field}) as {field}"
        return cast_map

    def get_merged_schema(self) -> dict:
        schema = dict()
        for f in self.files:
            file_schema = pl.read_parquet_schema(f)
            schema.update(file_schema)
        return schema
    
    def load_with_merge_schema(self, first_order_columns=["case_id"]) -> pl.LazyFrame:
        # union schema
        schema = self.get_merged_schema()

        lfs = []
        all_columns_sorted = first_order_columns + list(set(schema.keys()) - set(first_order_columns))

        for f in self.files:
            lf = pl.scan_parquet(f)
            missing_columns = set(all_columns_sorted) - set(lf.columns)

            # missin columns to null
            for c in missing_columns:
                lf = lf.with_columns(pl.lit(None).alias(c).cast(schema[c]))

            # ordring
            lf = lf.select(all_columns_sorted)
            lfs.append(lf)

        return pl.concat(lfs)

    def get_agg_data(self, lf: pl.LazyFrame ) -> pl.LazyFrame:
        temp_table_name = "tmp"
        field_agg_map = self.get_field_agg_map(lf)

        group_by_sql = f"""
                        SELECT case_id,
                        {','.join(field_agg_map.values())}
                        FROM {temp_table_name}
                        GROUP BY case_id    
                        """

        lf_agg = pl.SQLContext(frames={temp_table_name: lf}).execute(group_by_sql)
        return lf_agg

    def write_agg_data(self, lf:  pl.LazyFrame, path: str):
        df_agg = self.get_agg_data(lf)
        df_agg.write_parquet(path, compression="gzip")


loader = ParquetDataLoader("/home/w1/work/data/parquet_files/train")
df = loader.load_with_merge_schema()
df.sink_parquet("/home/w1/work/polars_merged.parquet",row_group_size=1000000)
df.collect(streaming=True).write_parquet("/home/w1/work/polars_merged.parquet")
