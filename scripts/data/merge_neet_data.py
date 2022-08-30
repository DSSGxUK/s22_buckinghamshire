"""
Concatenates different years of NEET data into one dataset 

1. Add column 'ccis_period_end' with YEAR-03-01 as the value

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the canonicalized neet csv. 
output : string
    The filepath to the output merged neet csv. 

Returns
---------
csv file
   merged neet dataset saved at output filepath as a csv file.

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
from src.constants import CCISDataColumns, NA_VALS

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input", required=True, help="where to find the input canonicalized ccis data csv"
)
parser.add_argument(
    "--output", required=True, help="where to put the output merged ccis csv"
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    neet_dfs = d.load_csvs(
        args.input,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger,
    )

    neet_dfs = {
        d.savedate_to_datetime(date) + relativedelta(months=1): df
        for date, df in neet_dfs.items()
    }
    merged_neet_df = pd.concat(
        [
            d.add_column(
                df, CCISDataColumns.ccis_period_end, datetime.strftime(date, "%Y-%m-%d")
            )
            for date, df in sorted(neet_dfs.items(), key=lambda x: x[0])
        ],
        axis=0,
    )

    csv_fp = args.output
    if args.debug:
        csv_fp = f.tmp_path(csv_fp)

    logger.info(f"Saving merged data for NEET data to {csv_fp}")
    merged_neet_df.to_csv(csv_fp, index=False)
