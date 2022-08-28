"""
1. Drop columns that seem to be inaccurate for absences
2. Compute absences based on codes
3. Add a column for the term type and end year

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input_att : string
    The filepath to the input attendance csv. 
input_school_info : string
    The filepath to the input school info csv
output : string
    The filepath to the output annotated attendance csv. 

Returns
---------
csv file
   annotated attendance dataset saved at output filepath as a csv file.
   
"""
import argparse
import pandas as pd
import numpy as np
import os
from pprint import pprint
import inflection
import re
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

# DVC Params
from src.constants import (
    AttendanceDataColumns,
    UNKNOWN_CODES,
    NA_VALS
)

# Other imports
from src import merge_utils as mu
from src import attendance_utils as a
from src import data_utils as d
from src import file_utils as f
from src import log_utils as l

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--input_att', required=True,
                    help='where to find the input attendance merged csv')
parser.add_argument('--input_school_info', required=True,
                    help='where to find the input school info csv')
parser.add_argument('--output', required=True,
                    help='where to put the output attendance annotated csv')

def add_absence_columns(df):
    """
    Add total absence columns for different types of absence reasons e.g. authorised, unauthorised and approved activities
    Calculate a total absences column = sum of all absence reasons for each student-term. 
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe with attendance data
    
    Returns
    --------
    df: pd.DataFrame
        dataframe with additional absence columns containing totals
    """
    authorised_columns = a.get_authorised_reason_columns(df)
    assert AttendanceDataColumns.termly_sessions_authorised not in authorised_columns
    unauthorised_columns = a.get_unauthorised_reason_columns(df)
    assert AttendanceDataColumns.termly_sessions_unauthorised not in unauthorised_columns
    approved_columns = a.get_approved_reason_columns(df)
    nonabsence_columns = a.get_nonabsence_reason_columns(df)
    
    # If there are no columns of the following type, we will set value of sum to NaN
    df = d.add_column(
        df, 
        AttendanceDataColumns.authorised_absences,
        d.to_int(df[authorised_columns]).sum(axis=1, min_count=1)
    )
    df = d.add_column(
        df, 
        AttendanceDataColumns.unauthorised_absences, 
        d.to_int(df[unauthorised_columns]).sum(axis=1, min_count=1)
    )
    df = d.add_column(
        df, 
        AttendanceDataColumns.approved_activities,
        d.to_int(df[approved_columns]).sum(axis=1, min_count=1)
    )
    df = d.add_column(
        df, 
        AttendanceDataColumns.total_absences, 
        d.to_int(df[[AttendanceDataColumns.authorised_absences, 
                     AttendanceDataColumns.unauthorised_absences, 
                     AttendanceDataColumns.approved_activities]]) \
                 .sum(axis=1, min_count=1)
    )
    df = d.add_column(
        df,
        AttendanceDataColumns.total_nonabsences,
        d.to_int(df[nonabsence_columns]).sum(axis=1, min_count=1)
    )
    
    return df

def term_end_to_term_type(term_ends):

    months = pd.to_datetime(term_ends).apply(lambda x: x.month)
    
    sum_month = 10  # October
    aut_month = 1  # January
    spr_month = 5  # May
    
    return months.replace({sum_month: 'summer', aut_month: 'autumn', spr_month: 'spring'})

def get_term_end_year(df):
    adjustments = {
        'summer': relativedelta(months=0),
        'spring': relativedelta(months=5),
        'autumn': relativedelta(months=9),
    }
    term_delta = df[AttendanceDataColumns.term_type].replace(adjustments)
    term_end_years = df[AttendanceDataColumns.term_end] + term_delta
    return term_end_years.apply(lambda x: x.year)


def add_term_columns(df):
    df = df.copy()
    df[AttendanceDataColumns.term_end] = pd.to_datetime(df[AttendanceDataColumns.term_end])
    df = d.add_column(df, AttendanceDataColumns.term_type, term_end_to_term_type(df[AttendanceDataColumns.term_end]))
    df = d.add_column(df, AttendanceDataColumns.year, get_term_end_year(df))
    return df
    

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    df = d.load_csv(
        args.input_att, 
        drop_empty=False, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger
    )

    school_df = d.load_csv(
        args.input_school_info, 
        drop_empty=False, 
        drop_single_valued=False, 
        drop_missing_upns=False, 
        drop_duplicates=False, 
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger
    )
    
    # Merge school info into attendance data
    logger.info('Merging school info with attendance data')
    df = mu.merge_priority_dfs(
        [school_df, df],  # The school df is the higher priority information
        on=AttendanceDataColumns.establishment_number,
        how='right',
        unknown_vals=UNKNOWN_CODES
    )
    # Drop termly sessions authorised, termly sessions unauthorised, present_am, present_pm columns
    logger.info('Dropping inconsistent columns')
    drop_cols = [AttendanceDataColumns.termly_sessions_authorised, AttendanceDataColumns.termly_sessions_unauthorised, AttendanceDataColumns.present_am, AttendanceDataColumns.present_pm]
    df = d.safe_drop_columns(df, drop_cols)
    # Add columns for authorised_absences, unauthorised_absences, approved_activities
    logger.info('Adding columns for absences')
    df = add_absence_columns(df)
    logger.info('Adding term type and end year columns')
    df = add_term_columns(df)
    
    csv_fp = args.output
    if args.debug:
         csv_fp = f.tmp_path(csv_fp)
    
    logger.info(f'Saving annotated data to {csv_fp}')
    df.to_csv(csv_fp, index=False)
