"""
This utils file contains helpers for merging dataframes
"""

from typing import Optional, Dict
from collections import defaultdict

import numpy as np
import pandas as pd

from .py_utils import remove_suffix


def merge_priority_data(
    df1, df2, on, how="outer", unknown_vals: Optional[Dict] = None, na_vals=tuple()
):
    if unknown_vals is None:
        unknown_vals = defaultdict(list)
    else:
        unknown_vals = defaultdict(list, unknown_vals)

    drop_key = "__drop"
    df = df1.merge(df2, how=how, on=on, suffixes=(None, drop_key))
    to_drop_cols = [col for col in df.columns if col.endswith(drop_key)]
    for col in to_drop_cols:
        orig_col = remove_suffix(col, drop_key)
        # orig_col = col.removesuffix(drop_key)
        # First fill na values with the other column
        new_col_values = df[col].astype(df[orig_col].dtype.name)
        df[orig_col] = df[orig_col].fillna(new_col_values)
        # If there are additional unknown vals try to fill it with known vals from the other df
        replacement_dict = {
            val: pd.NA for val in (list(na_vals) + list(unknown_vals[orig_col]))
        }
        # Replace unknown values with nan
        # fill it in with the second column of only known values
        # fill in any remaining nans with the original column and its unknown values
        df[orig_col] = (
            df[orig_col]
            .replace(replacement_dict)
            .fillna(new_col_values.replace(replacement_dict))
            .fillna(df[orig_col])
        )
    df = df.drop(to_drop_cols, axis=1)
    return df


def merge_priority_dfs(
    dfs, on, how="outer", unknown_vals: Optional[Dict] = None, na_vals=tuple()
):
    df = dfs[0]
    for other_df in dfs[1:]:
        df = merge_priority_data(
            df, other_df, on=on, how=how, unknown_vals=unknown_vals, na_vals=na_vals
        )
    return df
