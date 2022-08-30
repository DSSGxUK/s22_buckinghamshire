"""
This utils file contains helpers for aggregating data
"""
from typing import Callable, Dict, Iterable, Union
import logging
from functools import partial
import pandas as pd

from src.constants import CATEGORICAL_SEP, UNKNOWN_CODES, NA_VALS, Aggregations
from . import log_utils as l
from . import data_utils as d
from .error_utils import error_with_logging


def expand_categorical_max_agg_dict(
    agg_dict: Dict[str, Union[str, Callable]],
    columns: Iterable[str],
    prefix_sep: str = CATEGORICAL_SEP,
    logger: logging.Logger = l.PrintLogger,
):
    """
    TODO: Documentation

    Parameters
    -----------
    agg_dict : dict
    columns : iterable
    prefix_sep : string
    logger : logging.Logger

    Returns
    -----------

    TODO: Unit Test
    """
    categorical_max_aggs = [
        col
        for col, agg_type in agg_dict.items()
        if agg_type == Aggregations.categorical_max
    ]
    df_catcols = []
    for catcol in categorical_max_aggs:
        matching_catcols = [
            col for col in columns if col.startswith(catcol + prefix_sep)
        ]
        if len(matching_catcols) == 0:
            logger.warning(
                f"We cannot aggregate over categorical column {catcol} since did not occur in provided columns."
            )
        else:
            df_catcols += matching_catcols

    logger.debug(
        f"Replacing columns {categorical_max_aggs} in agg_dict with {df_catcols}."
    )
    return {
        **{catcol: Aggregations.max for catcol in df_catcols},
        **{k: v for k, v in agg_dict.items() if k not in categorical_max_aggs},
    }


def aggregate_last_with_unknown(x, unknown_codes, na_vals):
    """
    TODO: Documentation

    Parameters
    -----------

    Returns
    -----------

    TODO: Unit Test
    """
    if len(x) == 0:
        return pd.NA

    y = x[~d.isna(x, na_vals=na_vals)]
    z = y[~y.isin(unknown_codes)]
    if len(z) == 0:
        return y.iloc[-1]
    else:
        return z.iloc[-1]


def build_agg(
    agg: str,
    unknown_codes=UNKNOWN_CODES,
    na_vals=NA_VALS,
    logger: logging.Logger = l.PrintLogger,
):
    """Just turns a single aggregation type into the corresponding aggregation for use with pandas.
    Does not perform validation, assumes the input `agg` is a valid `Aggregations`.

    TODO: Documentation

    Parameters
    -----------

    Returns
    -----------

    TODO: Unit Test
    """
    if agg == Aggregations.last_with_unknown:
        return partial(
            aggregate_last_with_unknown, unknown_codes=unknown_codes, na_vals=na_vals
        )
    else:
        return agg  # All other aggregation types are just passed to pandas as strings.


def build_agg_dict(
    agg_dict: Dict[str, str],
    columns,
    unknown_codes=UNKNOWN_CODES,
    na_vals=NA_VALS,
    logger: logging.Logger = l.PrintLogger,
):
    """
    TODO: Documentation

    Parameters
    -----------

    Returns
    -----------

    TODO: Unit Test
    """
    # First do the aggs that are just replacements
    agg_dict = {col: build_agg(agg_type) for col, agg_type in agg_dict.items()}
    # These aggs only make sense for dictionary inputs
    agg_dict = expand_categorical_max_agg_dict(agg_dict, columns=columns, logger=logger)

    return agg_dict


def gby_agg_with_logging(
    df: pd.DataFrame,
    groupby_column,
    agg_dict: Dict[str, Union[str, Callable]],
    logger: logging.Logger = l.PrintLogger,
):
    """
    TODO: Documentation

    Parameters
    -----------

    Returns
    -----------

    TODO: Unit Test
    """
    logger.info(f"Aggregating columns {agg_dict}")
    prev_columns = set(df.columns)
    filtered_agg_dict = {k: v for k, v in agg_dict.items() if k in prev_columns}
    removed_from_agg_dict = agg_dict.keys() - prev_columns
    if len(removed_from_agg_dict) > 0:
        logger.warn(
            f"Could not aggregate on columns {removed_from_agg_dict} because they did not exist."
        )
    df = df.groupby(by=groupby_column).agg(filtered_agg_dict).reset_index()
    dropped_by_agg = list(prev_columns - set(df.columns))
    if len(dropped_by_agg) > 0:
        logger.warning(f"Aggregation implicitly dropped columns {dropped_by_agg}")

    return df
