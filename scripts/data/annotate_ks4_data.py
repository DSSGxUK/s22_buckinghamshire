"""
Annotate KS4 dataset by applying the following operations
1. Incorporate establishment information
2. Decode coded columns into names or numbers (e.g. Y/N -> 1/0) TODO

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input_ks4 : string
    The filepath to the input ks4 csv. 
input_school_info : string
    The filepath to the input school info csv.
output : string
    The filepath to the output annotated ks4 csv. 

Returns
-----------
csv file
   annotated ks4 dataset saved at output filepath as a csv file.  

"""
import pandas as pd
import os
import argparse
from datetime import datetime

# DVC Params
from src.constants import (
    NA_VALS,
    UNKNOWN_CODES,
    SchoolInfoColumns,
)

# Other code
from src import merge_utils as mu
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--input_ks4', required=True,
                    help='where to find the input ks4 merged csv')
parser.add_argument('--input_school_info', required=True,
                    help='where to find the input school info csv')
parser.add_argument('--output', required=True,
                    help='where to put the output ks4 annotated csv')

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    df = d.load_csv(
        args.input_ks4, 
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
    
    logger.info(f'Merging establishment information into ks4')
    df = mu.merge_priority_dfs(
        [school_df, df],  # The school df is the higher priority information
        on=SchoolInfoColumns.la_establishment_number,
        how='right',
        unknown_vals=UNKNOWN_CODES,
        na_vals=NA_VALS,
    )
    
    csv_fp = args.output
    if args.debug:
         csv_fp = f.tmp_path(csv_fp)
    
    logger.info(f'Saving annotated data to {csv_fp}')
    df.to_csv(csv_fp, index=False)
    