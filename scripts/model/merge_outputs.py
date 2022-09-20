"""
Combines final output csvs

Parameters
--input : string
    The filepath of where the predictions_orig.csv is located
--input_basic : string
    The filepath of where the predictions_basic.csv is located
--output : string
    The filepath of where to save the csv containing the predictions and risk scores for each student 

Returns
-----------
csv files
"""

import argparse
import pandas as pd
import pickle as pkl
import shap
from dataclasses import asdict

# need to remove this
from src import cv
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import merge_utils as mu

from src.constants import (
    CensusDataColumns,
    KSDataColumns,
    AttendanceDataColumns,
    CharacteristicsDataColumns,
    UPN,
    YEAR,
    NA_VALS,
    UNKNOWN_CODES,
    Targets,
)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")

parser.add_argument(
    "--output",
    type=lambda x: x.strip("'"),
    required=True,
    help="where to output the csv file containing predictions and data for power bi dashboard",
)
parser.add_argument(
    "--input",
    type=lambda x: x.strip("'"),
    required=True,
    help="where the input csv containing students for prediciting on is located",
)
parser.add_argument(
    "--input_basic",
    type=lambda x: x.strip("'"),
    required=True,
    help="where the input csv containing students for prediciting on is located",
)
if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    logger.info(f"Joining datasets {args.input} and {args.input_basic}")

    df = d.load_csv(
        args.input,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=False,
        read_as_str=False,
        drop_missing_upns=False,
        use_na=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger,
    )
    
    if "UPN" in df.columns :
        index_cols = ["UPN"]
    else :
        index_cols = [UPN] 
    #breakpoint()
    logger.info(f"Setting index cols {index_cols}")
    df.set_index(index_cols, inplace=True, drop=True)

    df_basic = d.load_csv(
        args.input_basic,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=False,
        read_as_str=False,
        drop_missing_upns=False,
        use_na=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger,
    )

    #index_cols = [UPN] if args.single else [UPN, YEAR]
    #logger.info(f"Setting index cols {index_cols}")
    df_basic.set_index(index_cols, inplace=True, drop=True)

    output_csv = pd.concat([df,df_basic],axis=0)
    #breakpoint()

    csv_fp = f.tmp_path(args.output, debug=args.debug)
    logger.info(f"Saving predictions to {csv_fp}")
    output_csv.to_csv(csv_fp, index=True)  # Index is True since the index is UPN




