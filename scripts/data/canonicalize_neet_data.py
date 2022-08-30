"""
Canonicalizes CCIS (NEET) dataset by applying the following operations
1. Drop completely duplicated rows
2. Rename columns to use snake_case
  - change UniquePupilNumber to upn and Estab to EstablishmentNumber
  - change 'moth/year of birth' to 'month_year_of_birth'
  - split up month/year of birth into two columns
3. Fix poor parsing of excel dates in feb18, feb21, and feb22
4. Write file back to processed data with a name of form neet_{date}.csv
5. Incorporate establishment information TODO
6. Decode columns with codes into new columns TODO

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the neet ccis csv. 
output : string
    The filepath to the output canonicalized neet ccis csv. 

Returns
---------
csv file
   canonicalized neet dataset saved at output filepath as a csv file.

"""
import pandas as pd
import os
import argparse

# DVC Params
from src.constants import CCIS_COLUMN_RENAME, CCISDataColumns, NA_VALS

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input", required=True, help="where to find the input ccis neet csv"
)
parser.add_argument(
    "--output",
    required=True,
    help="where to put the output canonicalized ccis neet csv",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    # Drop single-valued, empty columns, duplicate rows
    neet_dfs = d.load_csvs(
        args.input,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,
        na_vals=NA_VALS,
        logger=logger,
    )

    # Do column name rename
    logger.info("Renaming columns of NEET dataset")
    neet_dfs = {k: df.rename(columns=CCIS_COLUMN_RENAME) for k, df in neet_dfs.items()}
    logger.info("Removing bad values")
    neet_dfs = {k: df.replace("#VALUE!", "") for k, df in neet_dfs.items()}

    # Fix poor date parsing
    # TODO: use d.normalize_date function to fix all dates that are excel dates.
    neet_dfs["feb18"][CCISDataColumns.review_date] = neet_dfs["feb18"][
        CCISDataColumns.review_date
    ].apply(lambda x: d.read_excel_date(x) if not d.isna_scalar(x) else x)
    neet_dfs["feb18"][CCISDataColumns.neet_start_date] = neet_dfs["feb18"][
        CCISDataColumns.neet_start_date
    ].apply(lambda x: d.read_excel_date(x) if not d.isna_scalar(x) else x)

    csv_fps = args.output
    if args.debug:
        csv_fps = f.tmp_paths(csv_fps)

    for k, df in neet_dfs.items():
        csv_fp = csv_fps[k]
        logger.info(f"Saving canonicalized data for {k} to {csv_fp}")
        df.to_csv(csv_fp, index=False)
