"""
Canonicalizes attendance dataset by applying the following operations
1. Drop completely duplicated rows
2. Merge in upns from old data that don't appear in new data
3. Rename columns to use snake_case
  - rename coded columns to their names
  - drop code_f column
4. Incorporate establishment information TODO
5. Decode columns with codes into new columns TODO
6. Write file back to processed data with a name of form attendance_{date}.csv

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the attendance csv. 
output : string
    The filepath to the output canonicalized attendance csv. 

Returns
---------
csv file
   canonicalized attendance dataset saved at output filepath as a csv file.

"""
import pandas as pd
import os
import argparse
from datetime import datetime

# DVC Params
from src.constants import (
    ATTENDANCE_COLUMN_RENAME,
    NA_VALS
)

# Other code
from src import merge_utils as mu
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d


parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--input_new', required=True,
                    help='where to find the input new attendance csv')
parser.add_argument('--input_old', required=True,
                    help='where to find the input old attendance csv')
parser.add_argument('--output', required=True,
                    help='where to put the output canonicalized attendance csv')

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    # Drop duplicate rows
    att_dfs = d.load_csvs(
        args.input_new, 
        drop_empty=False, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger
    )
    old_att_dfs = d.load_csvs(
        args.input_old, 
        drop_empty=False, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger
    )
    
    # Merge in old data
    logger.info('Merging in old data from autumn and spring 2021')
    att_dfs['jan22'] = mu.merge_priority_dfs(
        [att_dfs['jan22'], old_att_dfs['aut21']],
        on=['UPN', 'EnrolStatus'],
        how='outer',
        unknown_vals=[],
        na_vals=NA_VALS,
    )
    att_dfs['may21'] = mu.merge_priority_dfs(
        [att_dfs['may21'], old_att_dfs['spr21']],
        on=['UPN', 'EnrolStatus'],
        how='outer',
        unknown_vals=[],
        na_vals=NA_VALS,
    )

    # Do column name rename
    logger.info('Renaming columns of attendance dataset')
    att_dfs = {k: df.rename(columns=ATTENDANCE_COLUMN_RENAME) for k, df in att_dfs.items()}
    logger.info('Removing empty column code_f from attendance dataset')
    att_dfs = {k: df.drop('code_f', axis=1) if 'code_f' in df.columns else df for k, df in att_dfs.items()}
    logger.info('Removing bad values')
    att_dfs = {k: df.replace('#VALUE!', '') for k, df in att_dfs.items()}
    
    csv_fps = args.output
    if args.debug:
         csv_fps = f.tmp_paths(csv_fps)
    
    for k, df in att_dfs.items():
        csv_fp = csv_fps[k]
        logger.info(f'Saving canonicalized data for {k} to {csv_fp}')
        df.to_csv(csv_fp, index=False)
    