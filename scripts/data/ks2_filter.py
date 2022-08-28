"""
Keeps the KS2 column data from the KS4 dataset. We do this as we will not have KS4 exam data at the time of predicting whether students are NEET (before Year 11). 

Parameters
----------
--debug: bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
--input: string (filepath)
    Filepath of the KS4 csv file. This is a required parameter.
--output: string (filepath)
    Filepath of where to save the output csv file. This is a required parameter. 

Returns
-------
csv file
    ks2 data ready to be merged with other datasets
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
from dataclasses import asdict

# DVC Params
from src.constants import (
    KS2Columns,
    UPN,
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
                    help='where to find input csv file: annotated ks4 data')
parser.add_argument('--output', required=True,
                    help='where to output the produced csv file')

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    df = d.load_csv(  # Modeling dataset so we drop unnecessary columns and entries
        args.input, 
        drop_empty=True, 
        drop_single_valued=True,
        drop_duplicates=True,
        read_as_str=True,  # This will ensure values are not cast to floats
        use_na=True,  # and this will ensure empty values are read as nan
        drop_missing_upns=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        logger=logger
    )
    
    logger.info(f'Initial row count {len(df)}')
    logger.info(f'Initial column count {len(df.columns)}')
    
    logger.info(f'Only keeping KS2 columns {asdict(KS2Columns).values()}')
    df = df[asdict(KS2Columns).values()]
    
    logger.info(f'Final row count {len(df)}')
    logger.info(f'Final column count {len(df.columns)}')
    
    csv_fp = args.output
    if args.debug:
         csv_fp = f.tmp_path(csv_fp)
    
    logger.info(f'Saving categorical data to {csv_fp}')
    df.to_csv(csv_fp, index=False)
    