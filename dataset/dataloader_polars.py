import os
import glob
import polars as pl

pl.Config.set_streaming_chunk_size(1000)

class ParquetDataLoader:
    def __init__(self, work_path="/home/w1/work", data_path: str = "/home/w1/work/data/parquet_files/train"):
        self.data_path = data_path
        self.base_files =  self.get_files_with_glob_pattern(f"{self.data_path}/*_base.parquet")
        self.files = self.get_files_with_glob_pattern(f"{self.data_path}/*.parquet") - self.base_files
        self.work_path = work_path

    def get_files_with_glob_pattern(self, pattern: str) -> set:
        return {os.path.abspath(file) for file in glob.glob(pattern)}

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

    def get_agg_data(self, lf: pl.LazyFrame) -> pl.LazyFrame:
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

    def aggregated_parquet(self,input_files,write_suffix="aggregated") -> list:
        written_files = []
        for f in input_files:
            # get file name
            file_name = os.path.basename(f).replace(".parquet", "")
            write_file_path = f"{self.work_path}/{file_name}_{write_suffix}.parquet"

            print(f"processing {f} -> {write_file_path}")
            lf = pl.scan_parquet(f)
            lf_agg = self.get_agg_data(lf)
            lf_agg.collect(streaming=True).write_parquet(write_file_path)
            written_files.append(write_file_path)
        return written_files

    def union_aggregated_files(self, topic: str) -> pl.LazyFrame:
        files = self.get_files_with_glob_pattern(f"{self.work_path}/*_{topic}*_aggregated.parquet")
        lf = pl.scan_parquet(list(files))
        lf = self.get_agg_data(lf)
        return lf
    
    def write_union_aggregated_files(self):
        write_file_pathes = []
        for topic in self.TOPICS:
            write_path = f"{self.work_path}/{topic}_unioned.parquet"
            print(f"processing union topic:{topic}")
            lf = self.union_aggregated_files(topic)
            lf.collect(streaming=True).write_parquet(write_path,use_pyarrow=True)
            write_file_pathes.append(write_path)
        return write_file_pathes
    
    def join_aggregated_files(self, files: list) -> pl.LazyFrame:
        #lazyframe join은 소용량밖에 처리 못함 버그,,
        base_df_path = None
        for i,f in enumerate(files):
            print(f"processing join {f}")
            if base_df_path:
                base_df = pl.read_parquet(base_df_path)
            else:
                base_df = pl.read_parquet(self.base_files)

            add_lf = pl.read_parquet(f,use_pyarrow=True)
            base_df = base_df.join(add_lf,on="case_id",how="left",suffix=f"_{i}")

            base_df_path = f"{self.work_path}/joined.parquet"
            base_df.write_parquet(base_df_path,use_pyarrow=True)
            print(f"saved : {base_df_path}")

    def join_agg_all_files(self):
        self.aggregated_parquet(self.files,write_suffix="aggregated")
        unioned_aggreegated_files = self.write_union_aggregated_files()
        self.join_aggregated_files(unioned_aggreegated_files)
        print("join complete")


loader = ParquetDataLoader( "/home/w1/work","/home/w1/work/data/parquet_files/train",)
loader.join_agg_all_files()
"""
loader = ParquetDataLoader( "/home/w1/work","/home/w1/work/data/parquet_files/train",)
loader.
"""
