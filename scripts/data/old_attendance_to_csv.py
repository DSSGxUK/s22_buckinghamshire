"""
Converts old attendance data excel file to a csv

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

import argparse

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d


parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input",
    required=True,
    help="where to find the input old attendance data excel file",
)
parser.add_argument(
    "--output", required=True, help="where to put the output attendance data csv"
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    csv_fp = f.tmp_path(args.output, debug=args.debug)

    d.save_xl_to_csv(xl_fp=args.input, csv_fp=csv_fp, sheet_name="Data", logger=logger)
