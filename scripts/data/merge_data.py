"""
Concatenates different years of attendance data into one dataset 

1. Add column 'term_end' to data
2. Merge datasets

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the canonicalized attendance csv. 
output : string
    The filepath to the output merged attendance csv. 

Returns
---------
csv file
   merged attendance dataset saved at output filepath as a csv file.

"""
import argparse
import pandas as pd
from datetime import datetime
from dataclasses import asdict

# DVC Params
from src.constants import (
    NA_VALS, 
    DatasetTypes, 
    DATA_DATE,
    AttendanceOriginalDataColumns,
    CCISOriginalDataColumns,
    CharacteristicsOriginalColumns,
    CensusDataOriginalColumns,
    KS2OriginalColumns,
    KS4_COLUMN_RENAME,
)

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--inputs",
    required=True,
    nargs="*",
    type=lambda x: x.strip("'"),
    help="where to find the input canonicalized data csvs",
)
parser.add_argument(
    "--data_dates",
    required=True,
    nargs="*",
    type=lambda x: x.strip("'"),
    help="The date of of the data submission in the form of [Month][Yr] (e.g. oct21 is for October 2021)",
)
parser.add_argument(
    "--output", required=True, 
    type=lambda x: x.strip("'"),
    help="where to put the output merged attendance csv"
)
parser.add_argument(
    "--dataset_type",
    required=True,
    type=lambda x: x.strip("'"),
    choices=asdict(DatasetTypes).values(),
    help="the type of dataset we are merging",
)


def merged_data_validation(df):
    assert not (
        d.isna(df[DATA_DATE]).any()
    ), f"Some data dates are missing. This is likely a bug in this script."
    try:
        pd.to_datetime(df[DATA_DATE], errors="raise")
    except ValueError as e:
        print(
            f"Parsing of dates from the {DATA_DATE} column failed. This is likely an bug in the code."
        )
        raise e

COLUMN_DICT = {
    DatasetTypes.attendance: asdict(AttendanceOriginalDataColumns).values(),
    DatasetTypes.census: asdict(CensusDataOriginalColumns).values(),
    DatasetTypes.ccis: asdict(CCISOriginalDataColumns).values(),
    DatasetTypes.ks4: set(KS4_COLUMN_RENAME.values()),
    DatasetTypes.characteristics: asdict(CharacteristicsOriginalColumns).values(),
    DatasetTypes.characteristics: asdict(KS2OriginalColumns).values(),
}

if __name__ == "__main__":
    args = parser.parse_args()
    #breakpoint()
    if len(args.inputs) != len(args.data_dates):
        raise ValueError(
            f"The number of input files must match the number of output files. You had {len(args.inputs)} input files and {len(args.source_annotations)} output files. The inputs were {args.inputs} and the outputs were {args.source_annotations}."
        )

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    try:
        data_dates = [d.savedate_to_datetime(date) for date in args.data_dates]
    except ValueError as e:
        print(
            f"Parsing of data dates from command line arguments failed. Please check that your data_dates are in the form [MONTH][YEAR] where [MONTH] is the three letter month abbreviation and [YEAR] is the two-digit year (e.g. 'sep21' for September 2021)."
        )
        raise e

    input_csv_dict = dict(zip(data_dates, args.inputs))
    dfs = d.load_csvs(
        input_csv_dict,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=False,  # Merging is not destructive
        read_as_str=True,
        na_vals=NA_VALS,
        use_na=True,
        logger=logger,
    )

    if len(dfs) == 0:
        logger.warning(f"No csvs passed, using default empty dataframe for type {args.dataset_type}")
        columns = list(set(COLUMN_DICT[args.dataset_type]) | {DATA_DATE,})
        merged_df = pd.DataFrame(columns=columns)
    else:
        logger.info("Adding column for data submission date of dataset")
        merged_df = pd.concat(
            [
                d.add_column(
                    df, DATA_DATE, datetime.strftime(date, "%Y-%m-%d"), inplace=True
                )
                for date, df in sorted(dfs.items(), key=lambda x: x[0])
            ],
            axis=0,
        )

    csv_fp = f.tmp_path(args.output, debug=args.debug)

    logger.info(f"Saving merged data to {csv_fp}")
    merged_df.to_csv(csv_fp, index=False)
