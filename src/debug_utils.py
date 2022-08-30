"""
This utils file contains helpers for debugging code.
"""

import pandas as pd
import numpy as np


def pd_print(x):
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(x)


def pd_display(x):
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(x)


def df_unique(df, filter_len=None, dropna=False):
    unique_values = {
        x: df[x].value_counts(dropna=dropna).sort_values(ascending=False)
        for x in df.columns
    }
    for x in df.columns:
        if all(isinstance(y, str) for y in unique_values[x].index):
            unique_values[x + "_len"] = (
                pd.Series(unique_values[x].index).apply(len).value_counts()
            )
    if filter_len is not None:
        for x in df.columns:
            if len(unique_values[x]) > filter_len:
                unique_values[x] = unique_values[x].iloc[:filter_len]
    return unique_values


def pd_debug():
    pd.set_option("display.max_rows", None, "display.max_columns", None)
