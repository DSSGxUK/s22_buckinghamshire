"""
This utils file contains helpers for processing the ccis dataset
"""

import pandas as pd

from .constants import CCISDataColumns, CharacteristicsDataColumns, NA_VALS
from . import data_utils as d


def split_up_birth_month_year(df, inplace=False):
    """
    Separates month and year of birth column into two separate columns: birth year and birth month

    Parameters
    -----------
    df : pd.DataFrame
        dataframe containing ccis data with month and year of birth column

    Returns
    ----------
    df : pd.DataFrame
        dataframe with separated birth month and year columns

    """
    if not inplace:
        df = df.copy()
    birth_data = pd.to_datetime(df[CCISDataColumns.month_year_of_birth])
    df[CCISDataColumns.birth_year] = birth_data.apply(lambda x: x.year).fillna(pd.NA)
    df[CCISDataColumns.birth_month] = birth_data.apply(lambda x: x.month).fillna(pd.NA)
    return d.safe_drop_columns(
        df, [CCISDataColumns.month_year_of_birth], inplace=inplace
    )

def compute_age(df):
    """
    Computes student age by beginning of autumn term based on birth month, birth year and current year.
    Handles NA values

    Parameters
    ----------
    df: pd.DataFrame
        A CCIS (NEET) dataframe with the `year`, `birth_year`, and `birth_month` columns of int-like type.

    Returns
    -------
    pd.Series
        A series containing the autumn term start age of the student in each row. If either the
        `year` value or `birth_year` or `birth_month` columns are NA, the returned age is also NA.
    """
    school_start_year = df[CharacteristicsDataColumns.year] - 1
    age = school_start_year - df[CharacteristicsDataColumns.birth_year]
    # Adjust age for those students born september or after
    age[df[CharacteristicsDataColumns.birth_month] >= 9] -= 1
    # Fill in NA for students whose birth month is unknown
    age[d.isna(df[CharacteristicsDataColumns.birth_month], na_vals=NA_VALS)] = pd.NA
    return age


# Deprecated - This was a very inefficient implementation of something simple
# def compute_age(df):
#     """
#     Computes student age by beginning of autumn term based on birth month, birth year and current year.
#     Handles NA values

#     Parameters
#     ----------
#     df: pd.DataFrame
#         A CCIS (NEET) dataframe with the `year`, `birth_year`, and `birth_month` columns of int-like type.

#     Returns
#     -------
#     pd.Series
#         A series containing the autumn term start age of the student in each row. If either the
#         `year` value or `birth_year` or `birth_month` columns are NA, the returned age is also NA.
#     """
#     school_start = pd.to_datetime(
#         (df[CharacteristicsDataColumns.year] - 1).astype(pd.StringDtype()) + "-08-31"
#     )  # gets the date of the end of the previous school year
#     school_start.name = "end"
#     birth_date = pd.to_datetime(
#         df[CharacteristicsDataColumns.birth_year].astype(pd.StringDtype())
#         + "-"
#         + df[CharacteristicsDataColumns.birth_month].astype(pd.StringDtype())
#         + "-01"
#     )  # gets the birth date and month
#     birth_date.name = "start"
#     end_start = pd.concat([school_start, birth_date], axis=1)
#     # Filter out the NA values and incorporate them back in with a `reindex_like`
#     nonempty_mask = end_start.notna().all(axis=1)
#     end_start_masked = end_start.loc[nonempty_mask, :]
#     if len(end_start_masked) == 0:
#         # There are no nonmissing entries
#         return d.empty_series(
#             len(end_start), name=CCISDataColumns.age, index=end_start.index
#         ).astype(pd.Int16Dtype())
#     return (
#         end_start_masked.apply(
#             lambda r: relativedelta(r["end"], r["start"]).years, axis=1
#         )
#         .reindex_like(end_start)
#         .astype(pd.Int16Dtype())
#     )


def neet_or_unknown(df):
    """
    Returns rows that have a neet OR unknown activity code. Handles pd.NA values.

    Parameters
    ----------
    df: pd.DataFrame
        A CCIS dataset with activity_code.

    Returns
    -------
    pd.Series
        A boolean series that is True if the student's activity is a NEET or "Unknown" activity
        and False otherwise. If the activity_code is missing (i.e. pd.NA), then
        the entry in the output is also pd.NA

    """
    # All codes 540 or above are NEET or unknown
    activity_codes = df[CCISDataColumns.activity_code].astype(pd.Int64Dtype())
    return df[CCISDataColumns.activity_code] >= 540


def unknown(df):
    """
    Returns rows that have an unknown activity code. Handles pd.NA values.

    Parameters
    ----------
    df: pd.DataFrame
        A CCIS dataset with activity_code.

    Returns
    -------
    pd.Series
        A boolean series that is True if the student's activity is a "Unknown" activity
        and False otherwise. If the activity_code is missing (i.e. pd.NA), then
        the entry in the output is also pd.NA
    """
    # All codes 810 and above are unknown
    activity_codes = df[CCISDataColumns.activity_code].astype(pd.Int64Dtype())

    return 800 <= activity_codes


def neet(df):
    """
    Returns rows that have a neet activity code. Handles pd.NA values.

    Parameters
    ----------
    df: pd.DataFrame
        A CCIS dataset with activity_code.

    Returns
    -------
    pd.Series
        A boolean series that is True if the student's activity is a NEET activity
        and False otherwise. If the activity_code is missing (i.e. pd.NA), then
        the entry in the output is also pd.NA
    """
    # All codes 540 or above and below 700 are NEET
    activity_codes = df[CCISDataColumns.activity_code].astype(pd.Int64Dtype())

    return (activity_codes >= 540) & (700 > activity_codes)


def in_compulsory_school(df):
    """
    Returns rows that have a compulsory school age activity code. Handles pd.NA values.

    Parameters
    ----------
    df: pd.DataFrame
        A CCIS dataset with activity_code.

    Returns
    -------
    pd.Series
        A boolean series that is True if the student's activity is a compulsory school activity
        and False otherwise. If the activity_code is missing (i.e. pd.NA), then
        the entry in the output is also pd.NA

    """
    # All codes below 200 are in compulsory school
    activity_codes = df[CCISDataColumns.activity_code].astype(pd.Int64Dtype())

    return (activity_codes < 200) & (activity_codes >= 0)
