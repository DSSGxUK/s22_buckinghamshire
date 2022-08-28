"""
Canonicalizes School Census dataset by applying the following operations
1. Drop completely duplicated rows
2. Rename columns to use snake_case
  - change Estab to establishment_number
  - Fix sen need column duplication
3. Incorporate establishment information TODO
4. Decode columns with codes into new columns TODO
5. Write file back to processed data with a name of form census_{date}.csv

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the census csv. 
output : string
    The filepath to the output canonicalized census csv. 

Returns
---------
csv file
   canonicalized census dataset saved at output filepath as a csv file.

"""
import pandas as pd
import os
import argparse
from datetime import datetime

# DVC Params
from src.constants import (
    SCHOOL_CENSUS_COLUMN_RENAME,
    NA_VALS
)

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d


parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--input', required=True,
                    help='where to find the input census csv')
parser.add_argument('--output', required=True,
                    help='where to put the output canonicalized census csv')

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    # Drop single-valued, empty columns, duplicate rows
    census_dfs = d.load_csvs(
        args.input, 
        drop_empty=False, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger
    )

    
    # Do column name rename
    logger.info('Renaming columns of census dataset')
    census_dfs = {k: df.rename(columns=SCHOOL_CENSUS_COLUMN_RENAME) for k, df in census_dfs.items()}
    logger.info('Removing bad values')
    census_dfs = {k: df.replace('#VALUE!', '') for k, df in census_dfs.items()}
    
    csv_fps = args.output
    if args.debug:
         csv_fps = f.tmp_paths(csv_fps)
    
    for k, df in census_dfs.items():
        csv_fp = csv_fps[k]
        logger.info(f'Saving canonicalized data for {k} to {csv_fp}')
        df.to_csv(csv_fp, index=False)
    