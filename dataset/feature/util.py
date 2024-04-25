from typing import List
from joblib import Parallel, delayed
import numpy as np
import polars as pl


def optimize_int_datatype(c_min: float, c_max: float):
    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
        return pl.Int8
    if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
        return pl.Int16
    if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
        return pl.Int32
    if c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
        return pl.Int64
    raise OverflowError('[-] not supported int-range')


def optimize_dataframe_datatype(col_type, c_min: float, c_max: float):
    c_min = c_min if c_min is not None else 0
    c_max = c_max if c_max is not None else 0
    if str(col_type)[:3] == 'Int':
        return optimize_int_datatype(c_min, c_max)
    try:
        np.array([c_min, c_max]).astype(np.float32)
        return pl.Float32
    except Exception as err:
        return pl.Utf8


def optimize_dataframe(df: pl.DataFrame, verbose=False) -> pl.DataFrame:
    start_memory: float = df.estimated_size('mb')
    data_types: List = Parallel(n_jobs=-1)(
        delayed(optimize_dataframe_datatype)(
            str(df[col].dtype), df[col].min(), df[col].max()
        )
        for col in df.columns
    )
    for col, data_type in zip(df.columns, data_types):
        df = df.with_columns(
            pl.col(col).cast(data_type),
        )
    end_memory: float = df.estimated_size('mb')
    if verbose:
        print(f'[*] Memory usage of dataframe is {start_memory:.4f} MB')
        print(f'[*] Memory usage after optimization is: {end_memory:.4f} MB')
        print(
            f'[+] Decreased by {100 * (start_memory - end_memory) / start_memory:.4f}%'
        )

    return df
