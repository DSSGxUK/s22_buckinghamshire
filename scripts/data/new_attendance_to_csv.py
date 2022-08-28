"""
Converts attendance data excel file to a csv

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the attendance data excel file. 
output : string
    The filepath to the output attendance data csv. 

Returns
---------
csv file
   attendance dataset saved at output filepath as a csv file.

"""
import pandas as pd
import os
import argparse
from datetime import datetime
import logging

# DVC Params
#from src.params import (
#)

# Other code
from src import file_utils as f
from src import log_utils as l

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--input', required=True,
                    help='where to find the input attendance data excel file')
parser.add_argument('--output', required=True,
                    help='where to put the output attendance data csv')

def sheet_name_to_datetime(date):
    return datetime.strptime(date, '%b %y')

def to_savedate(datetime_obj):
    return datetime_obj.strftime('%b%y').lower()

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    # Run logic
    att_excel_files = {k: pd.ExcelFile(v) for k, v in args.input.items()}
    
    att_dfs = {}
    for k, xl in att_excel_files.items():
        for sheet in xl.sheet_names:
            logger.info(f'Reading sheet {sheet} from excel file {args.input[k]}')
            if sheet == 'Intro':
                logger.info(f'Sheet name is Intro, so skipping')
                continue
            att_dfs[sheet_name_to_datetime(sheet)] = pd.read_excel(
                xl,
                sheet,
                keep_default_na=False, 
                na_values=[], 
                dtype=str, 
                header=3
            )
    for term, df in att_dfs.items():
        savedate = to_savedate(term)
        save_fp = args.output[savedate]
        if args.debug:
            save_fp = f.tmp_path(save_fp)
        logger.info(f'Saving {savedate} df to {save_fp}')
        df.to_csv(save_fp, index=False)
    