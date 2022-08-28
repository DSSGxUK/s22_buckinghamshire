"""
Concatenates different years of ks4 data into one dataset 

1. Add column 'ks4_period_end' to data
2. Merge datasets

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the canonicalized ks4 csv. 
output : string
    The filepath to the output merged ks4 csv. 

Returns
---------
csv file
   merged ks4 dataset saved at output filepath as a csv file.

"""
import argparse
import pandas as pd
import numpy as np
import os
from pprint import pprint
import inflection
import re
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta

# DVC Params
from src.constants import (
    KSDataColumns,
    NA_VALS,
)

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--input', required=True,
                    help='where to find the input canonicalized ks4 data csv')
parser.add_argument('--output', required=True,
                    help='where to put the output merged ks4 csv')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    ks4_dfs = d.load_csvs(
        args.input, 
        drop_empty=False, 
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger
    )
    
    # August 31 is the last day a student can take KS4
    ks4_dfs = {datetime.strptime('-'.join([yr, '08', '31']), '%Y-%m-%d'): df for yr, df in ks4_dfs.items()}
    logger.info('Adding column for end of ks4 period of each ks4 dataset')
    merged_ks4_df = pd.concat(
        [d.add_column(df, KSDataColumns.ks4_period_end, datetime.strftime(date, '%Y-%m-%d')) for date, df in sorted(ks4_dfs.items(), key=lambda x: x[0])], 
        axis=0
    )
    
    csv_fp = args.output
    if args.debug:
        csv_fp = f.tmp_path(csv_fp)
    
    logger.info(f'Saving merged data for KS4 data to {csv_fp}')
    merged_ks4_df.to_csv(csv_fp, index=False)