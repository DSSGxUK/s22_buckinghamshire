"""
Annotate School Census dataset by applying the following operations
1. Incorporate establishment information
2. Decode coded columns into names or numbers (e.g. Y/N -> 1/0) 

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input_census : string
    The filepath to the input census csv. 
input_school_info : string
    The filepath to the input school info csv
output : string
    The filepath to the output annotated census csv. 

Returns
----------------
csv file
   annotated census dataset saved at output filepath as a csv file.  

"""
import pandas as pd
import os
import argparse
from datetime import datetime

# DVC Params
from src.constants import (
    UNKNOWN_CODES,
    NA_VALS,
    CensusDataColumns,
)

# Other code
from src import merge_utils as mu
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

def remove_suffix(s,suffix):
    if s.endswith(suffix):
        s=s[:-len(suffix)]
    return s

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--input_census', required=True,
                    help='where to find the input census merged csv')
parser.add_argument('--input_school_info', required=True,
                    help='where to find the input school info csv')
parser.add_argument('--output', required=True,
                    help='where to put the output census annotated csv')

def merged_census_validation(df):
    fsme_values = set(df[CensusDataColumns.fsme_on_census_day].unique())
    fsme_values_expected = {'FALSE', 'TRUE'}
    assert  fsme_values== fsme_values_expected, f'fsme_on_census_day has extra values {fsme_values - fsme_values_expected}. Only  values {fsme_values_expected} are allowed. Please correct.'

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    df = d.load_csv(
        args.input_census, 
        drop_empty=False, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger
    )
    logger.info('Doing some validation on the incoming merged census df')
    merged_census_validation(df)

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
    
    logger.info(f'Initial row count {len(df)}')
    logger.info(f'Initial column count {len(df.columns)}')
    
    logger.info(f'Merging establishment information into census')
    df = mu.merge_priority_dfs(
        [school_df, df],  # The school df is the higher priority information
        on=CensusDataColumns.establishment_number,
        how='right',
        unknown_vals=UNKNOWN_CODES,
        na_vals=NA_VALS,
    )
    logger.info(f'Adding column for end_year of school term')
    df[CensusDataColumns.year] = pd.to_datetime(df[CensusDataColumns.census_period_end]).apply(lambda x: x.year)
    logger.info(f'Removing + at the end of ages')
    df[CensusDataColumns.age] = df[CensusDataColumns.age].apply(lambda x: remove_suffix(x,'+'))#x.removesuffix('+'))
    
    logger.info(f'Converting census column {CensusDataColumns.fsme_on_census_day} to a binary column')
    # There shouldn't be any na values
    logger.debug(f'{CensusDataColumns.fsme_on_census_day} currently has values {df[CensusDataColumns.fsme_on_census_day].unique()}')
    df[CensusDataColumns.fsme_on_census_day] = df[CensusDataColumns.fsme_on_census_day].replace({'TRUE': 1, 'FALSE': 0}).astype(int)
    logger.debug(f'{CensusDataColumns.fsme_on_census_day} now has values {df[CensusDataColumns.fsme_on_census_day].unique()}')
    
    logger.info(f'Final row count {len(df)}')
    logger.info(f'Final column count {len(df.columns)}')
    
    csv_fp = args.output
    if args.debug:
         csv_fp = f.tmp_path(csv_fp)
    
    logger.info(f'Saving annotated data to {csv_fp}')
    df.to_csv(csv_fp, index=False)
    