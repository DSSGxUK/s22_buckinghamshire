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
input : string
    The filepath to the input attendance csv. 
school_info : string
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
parser.add_argument('--input', required=True,
                    help='where to find the input attendance merged csv')
parser.add_argument('--school_info', required=True,
                    help='where to find the input school info csv')
parser.add_argument('--output', required=True,
                    help='where to put the output attendance annotated csv')

def add_absence_columns(df, inplace=False):
    """
    Add total absence columns for different types of absence reasons e.g. authorised, unauthorised and approved activities
    Calculate a total absences column = sum of all absence reasons for each student-term. 
    
    Parameters
    ----------
    df : pd.DataFrame
        dataframe with attendance data
    inplace: bool
        whether to do the operation in-place (this will mutate df, but save memory)
    
    Returns
    --------
    df: pd.DataFrame
        dataframe with additional absence columns containing totals
    """
    authorised_columns = a.get_authorised_reason_columns(df)
    assert AttendanceDataColumns.termly_sessions_authorised not in authorised_columns # This is not an "absence reason"
    unauthorised_columns = a.get_unauthorised_reason_columns(df)
    assert AttendanceDataColumns.termly_sessions_unauthorised not in unauthorised_columns # This is not an "absence reason"
    approved_columns = a.get_approved_reason_columns(df)
    nonabsence_columns = a.get_nonabsence_reason_columns(df)
    
    # If there are no columns of the following type, we will set value of sum to pd.NA
    df = d.add_column(
        df, 
        AttendanceDataColumns.authorised_absences,
        df[authorised_columns].astype(pd.Int16Dtype()).sum(axis=1, min_count=1),
        inplace=inplace
    )
    df = d.add_column(
        df, 
        AttendanceDataColumns.unauthorised_absences, 
        df[unauthorised_columns].astype(pd.Int16Dtype()).sum(axis=1, min_count=1),
        inplace=inplace
    )
    df = d.add_column(
        df, 
        AttendanceDataColumns.approved_activities,
        df[approved_columns].astype(pd.Int16Dtype()).sum(axis=1, min_count=1),
        inplace=inplace
    )
    df = d.add_column(
        df, 
        AttendanceDataColumns.total_absences, 
        df[[AttendanceDataColumns.authorised_absences, 
            AttendanceDataColumns.unauthorised_absences, 
            AttendanceDataColumns.approved_activities]] \
                .astype(pd.Int16Dtype())
                .sum(axis=1, min_count=1),
        inplace=inplace
    )
    df = d.add_column(
        df,
        AttendanceDataColumns.total_nonabsences,
        df[nonabsence_columns].astype(pd.Int16Dtype()).sum(axis=1, min_count=1),
        inplace=inplace
    )
    
    return df

def term_end_to_term_type(term_ends):

    # We know that months will not contain any NA values since it is coming from "data_date" which will always be specified when we
    # merge attendance data. This was validated by attendance_merged_validation
    months = pd.to_datetime(term_ends).apply(lambda x: x.month).astype(pd.Int16Dtype())
    
    sum_month = 10  # October
    aut_month = 1  # January
    spr_month = 5  # May
    
    return months.astype(pd.StringDtype()).replace({str(sum_month): 'summer', str(aut_month): 'autumn', str(spr_month): 'spring'})

def get_term_end_year(term_types, term_ends):
    adjustments = {
        'summer': relativedelta(months=0),
        'spring': relativedelta(months=5),
        'autumn': relativedelta(months=9),
    }
    term_delta = term_types.astype(object).replace(adjustments)
    
    term_end_years = term_ends + term_delta
    return term_end_years.apply(lambda x: x.year).astype(pd.Int16Dtype())


def add_term_columns(df, inplace=False):
    if not inplace:
        df = df.copy()

    # term_end will never have NA values because data_date cannot have NA values. We raise an error
    # if there are any values in data_date that cannot be parsed
    df[AttendanceDataColumns.term_end] = pd.to_datetime(df[AttendanceDataColumns.data_date], errors='raise')  # Attendance data date corresponds to the term end date

    df = d.add_column(
        df, 
        AttendanceDataColumns.term_type, 
        term_end_to_term_type(df[AttendanceDataColumns.term_end]), 
        inplace=inplace
    )
    
    df = d.add_column(
        df, 
        AttendanceDataColumns.year, 
        get_term_end_year(term_types=df[AttendanceDataColumns.term_type], term_ends=df[AttendanceDataColumns.term_end]),
        inplace=inplace
    )
    
    return df

def attendance_merged_validation(df: pd.DataFrame):
    assert df[AttendanceDataColumns.data_date].isna().any() == False, "data_date column of attendance merged dataframe has null values, please ensure there is no error in your input to attendance_merged.py or any bug in that code."

    
if __name__ == '__main__':
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    df = d.load_csv(
        args.input,
        drop_empty=False, 
        drop_single_valued=False,
        drop_duplicates=False,  # Annotating is nondestructive
        read_as_str=True,
        na_vals=NA_VALS,
        use_na=True,
        logger=logger
    )
    attendance_merged_validation(df)

    school_df = d.load_csv(
        args.school_info, 
        drop_empty=False, 
        drop_single_valued=False, 
        drop_missing_upns=False, 
        drop_duplicates=False, 
        read_as_str=True,
        na_vals=NA_VALS,
        use_na=True,
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
    
    # Add columns for authorised_absences, unauthorised_absences, approved_activities
    logger.info('Adding columns for absences')
    df = add_absence_columns(df, inplace=True)

    logger.info('Adding term type and end year columns')
    df = add_term_columns(df, inplace=True)
    
    csv_fp = f.tmp_path(args.output, debug=args.debug)
    
    logger.info(f'Saving annotated data to {csv_fp}')
    df.to_csv(csv_fp, index=False)
