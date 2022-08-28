"""
Annotates NEET CCIS dataset
TODO: Test

1. Add school establishment info to the datasets merging on la_establishment_number
2. Add a column for NEET and for Unknown
3. Split up month and year of birth into separate columns
4. Add year column for joining datasets
5. Add column with student age 
6. Propagate NEET and unknowns to other instances of same upn
7. Convert columns with 'Y'&'N' data to 0 & 1

Rewrite codes to true names in new column. E.g. create a column 'activity_name' that has the name corresponding to the code in 'activity_code'. TODO 

Validated in 0019.0-am-debug-annotate-neet.ipynb

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the input ccis-neet csv. 
school_info : string
    The filepath to the input school info csv.
output : string
    The filepath to the output annotated ccis-neet csv. 
no_logging : bool
    Whether to turn off file logging - useful for tests

Returns
---------
csv file
   annotated neet dataset saved at output filepath as a csv file.

"""
import pandas as pd
import os
import argparse
from datetime import datetime
import re
from dateutil.relativedelta import relativedelta

from src.constants import (
    UNKNOWN_CODES,
    CCISDataColumns,
    NA_VALS
)

# Other code
from src import merge_utils as mu
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import ccis_utils as nu

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--input', required=True,
                    help='where to find the input CCIS merged data csv')
parser.add_argument('--school_info', required=True,
                    help='where to find the input canonicalized secondary school info csv')
parser.add_argument('--output', required=True,
                    help='where to put the output annotated CCIS data csv')
parser.add_argument('--no_file_logging', action='store_true',
                    help='turn off file logging - useful for tests')
    
def split_up_birth_month_year(df):
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
    df = df.copy()
    birth_data = pd.to_datetime(df[CCISDataColumns.month_year_of_birth])
    df[CCISDataColumns.birth_year] = birth_data.apply(lambda x: x.year)
    df[CCISDataColumns.birth_month] = birth_data.apply(lambda x: x.month)
    return df.drop(CCISDataColumns.month_year_of_birth, axis=1)
    
def annotated_neet_data_validation(df):
    assert not (d.isna(df[CCISDataColumns.year]).any()), 'Some year values are not filled for the CCIS data. This error likely occured during merging of CCIS datasets.'
    assert not (d.isna(df[CCISDataColumns.age]).any()), 'Some age values are not filled for the CCIS data. Please check all students have birth months and birth years filled in.'
    # TODO add validation for neet_ever and unknown_ever columns
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug, file_logging = (not args.no_file_logging))
    
    df = d.load_csv(
        args.input,
        drop_empty=False, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger
    )
    
    school_df = d.load_csv(
        args.school_info, 
        drop_empty=False, 
        drop_single_valued=False, 
        drop_missing_upns=False, 
        drop_duplicates=False, 
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger
    )
    
    # Merge school info into neet data
    logger.info('Merging school info with NEET data')
    df = mu.merge_priority_dfs(
        [school_df, df],  # The school df is the higher priority information
        on=CCISDataColumns.la_establishment_number,
        how='right',
        unknown_vals=UNKNOWN_CODES,
        na_vals=NA_VALS,
    )
    # Add column for neet and unknown
    logger.info('Adding columns for neet and unknown status')
    df = d.add_column(df, CCISDataColumns.neet, nu.neet(df).astype(int))
    df = d.add_column(df, CCISDataColumns.unknown, nu.unknown(df).astype(int))
    df = d.add_column(df, CCISDataColumns.compulsory_school, nu.in_compulsory_school(df).astype(int))
    
    # Split up month/year of birth into two columns
    logger.info('Splitting up birth month and birth year into separate columns')
    df = split_up_birth_month_year(df)
    
    logger.info('Adding year column for joining datasets')
    df[CCISDataColumns.year] = pd.to_datetime(df[CCISDataColumns.ccis_period_end]).apply(lambda x: x.year)
    
    logger.info('Compute age of students at the end of prior august')
    df[CCISDataColumns.age] = nu.compute_age(df)
    
    # Propogate neet and unknown statuses to other instances of the upn
    logger.info('Marking NEET and unknown students')
    df.loc[:, [CCISDataColumns.neet, CCISDataColumns.unknown]] = df[[CCISDataColumns.neet, CCISDataColumns.unknown]].astype(int)
    neet_propogated_df = df[[CCISDataColumns.upn, CCISDataColumns.neet, CCISDataColumns.unknown]] \
        .groupby(by=CCISDataColumns.upn) \
        .max() \
        .reset_index()
    df = df.merge(neet_propogated_df, on=CCISDataColumns.upn, how='left', suffixes=(None, '_ever'))
    
    # NEET data
    y_n_columns = [
        CCISDataColumns.send_flag,  # Y/N
        CCISDataColumns.currency_lapsed,  # Y/N
        CCISDataColumns.youth_contract_indicator,
        CCISDataColumns.sen_support_flag
    ]
    logger.info(f'Converting Y/N to 1/0 in columns {y_n_columns}')
    for col in y_n_columns:
        df[col] = df[col].replace({'Y':1,'N':0})
    
    # Some data validation
    logger.info('Validating annotated data')
    annotated_neet_data_validation(df)
    
    csv_fp = f.tmp_path(args.output, debug=args.debug)
    
    logger.info(f'Saving annotated data to {csv_fp}')
    df.to_csv(csv_fp, index=False)
