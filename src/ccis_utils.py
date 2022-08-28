"""
This utils file contains helpers for processing the ccis dataset
"""

import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from .constants import (
    CCISDataColumns,
)
from . import data_utils as d

def compute_age(df):
    """
    Computes student age based on birth month, birth year and current year 
    
    """
    school_start = pd.to_datetime((df[CCISDataColumns.year] - 1).astype(str) + '-08-31') # gets the date of the end of the previous school year
    school_start.name = 'end'
    birth_date = pd.to_datetime(df[CCISDataColumns.birth_year].astype(str) + '-' + df[CCISDataColumns.birth_month].astype(str) + '-01') #gets the birth date and month
    birth_date.name = 'start'
    return pd.concat([school_start, birth_date], axis=1).apply(lambda r: relativedelta(r['end'], r['start']).years, axis=1)

def neet_or_unknown(df):
    """
    Returns rows that have a neet OR unknown activity code     
    
    """
    # All codes 540 or above are NEET or unknown
    assert pd.api.types.is_numeric_dtype(df[CCISDataColumns.activity_code])
    return (df[CCISDataColumns.activity_code] >= 540 )

def unknown(df):
    """
    Returns rows that have an unknown activity code     
    """    
    # All codes 810 and above are unknown
    activity_codes = df[CCISDataColumns.activity_code].copy()
    activity_codes[d.isna(activity_codes)] = '-1'
    activity_codes = activity_codes.astype(int)
    
    return (800 <= activity_codes)

def neet(df):
    """
    Returns rows that have a neet activity code     
    
    """
    # All codes 540 or above and below 700 are NEET
    activity_codes = df[CCISDataColumns.activity_code].copy()
    activity_codes[d.isna(activity_codes)] = '-1'
    activity_codes = activity_codes.astype(int)
    
    return (activity_codes >= 540) & (700 > activity_codes)

def in_compulsory_school(df):
    """
    Returns rows that have a compulsory school age activity code     
    
    """    
    # All codes below 200 are in compulsory school
    activity_codes = df[CCISDataColumns.activity_code].copy()
    activity_codes[d.isna(activity_codes)] = '-1'
    activity_codes = activity_codes.astype(int)
    
    return (activity_codes < 200 ) & (activity_codes >= 0)
