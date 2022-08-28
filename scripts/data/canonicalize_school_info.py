"""
Canonicalize school info 

1. Strip the last rows with the messages about electoral wards and the 7777 school
2. Rename the columns to use snakecase
3. Add a row for the 7777 school
4. Parse the human readable list of electoral wards into a list delimited by ':'
5. Remove the "- The" at the end of some school names

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the school info csv. 
output : string
    The filepath to the output canonicalized school info csv. 

Returns
---------
csv file
   canonicalized school info dataset saved at output filepath as a csv file.

"""
import pandas as pd
import os
import argparse
from datetime import datetime
import re

# DVC Params
from src.params import (
    SCHOOL_INFO_RENAME,
    NA_VALS,
    SchoolInfoColumns
)

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--input', required=True,
                    help='where to find the input school info csv')
parser.add_argument('--output', required=True,
                    help='where to put the output canonicalized school info csv')


def parse_human_list(msg):
    re_split = r'(, )|( and )'
    re_remove = r'\d sites: '
    msg = re.sub(re_remove, '', msg)
    msg = re.sub(re_split, ':', msg)

    return msg

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    schools_df = d.load_csv(
        args.input, 
        drop_empty=False, 
        drop_single_valued=False, 
        drop_missing_upns=False, 
        drop_duplicates=False, 
        na_vals=NA_VALS,
        read_as_str=True,
    )
    
    # Strip the last rows with the message
    can_schools_df = schools_df.iloc[:37]
    # Rename the columns
    can_schools_df = can_schools_df.rename(columns=SCHOOL_INFO_RENAME)
    # Add the special school
    new_row = pd.DataFrame([{
        SchoolInfoColumns.la_establishment_number: '8257777', 
        SchoolInfoColumns.establishment_number: '7777', 
        SchoolInfoColumns.establishment_name: 'Other Special', 
        SchoolInfoColumns.establishment_type: 'Special',
        SchoolInfoColumns.establishment_status: '',
        SchoolInfoColumns.establishment_area: '',
        SchoolInfoColumns.establishment_postcode: '',
        SchoolInfoColumns.establishment_electoral_wards: ''
    }])
    assert set(new_row.keys()) == set(can_schools_df.columns)
    can_schools_df = pd.concat([can_schools_df, new_row], axis=0).reset_index(drop=True)
    # Parse the list of electoral wards
    can_schools_df[SchoolInfoColumns.establishment_electoral_wards] = can_schools_df[SchoolInfoColumns.establishment_electoral_wards].apply(lambda x: parse_human_list(x))
    # Remove the "- The" at the end of some school names
    can_schools_df[SchoolInfoColumns.establishment_name] = can_schools_df[SchoolInfoColumns.establishment_name].apply(lambda x: x.replace(' - The', ''))
    
    csv_fp = args.ouput
    if args.debug:
         csv_fp = f.tmp_path(csv_fp)
    
    logger.info(f'Saving canonicalized data for secondary schools to {csv_fp}')
    can_schools_df.to_csv(csv_fp, index=False)
    