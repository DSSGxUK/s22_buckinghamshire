"""
Convert neet ccis data excel file to a csv

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the neet ccis data excel file. 
output : string
    The filepath to the output neet ccis data csv. 

Returns
---------
csv file
   neet ccis dataset saved at output filepath as a csv file.

"""
import pandas as pd
import os
import argparse
from datetime import datetime

# DVC Params
# from src.params import

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d


parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input", required=True, help="where to find the input neet ccis data excel file"
)
parser.add_argument(
    "--output", required=True, help="where to put the output neet ccis data csv"
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    csv_fps = args.output
    if args.debug:
        csv_fps = f.tmp_paths(csv_fps)

    d.save_xls_to_csv(xl_fps=args.input, csv_fps=csv_fps, logger=logger)
