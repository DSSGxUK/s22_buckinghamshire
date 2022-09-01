"""
Annotates Characteristics dataset

1. Add school establishment info to the datasets merging on la_establishment_number
3. Split up month and year of birth into separate columns
4. Add year column for joining datasets
5. Add column with student age 
7. Convert columns with 'Y'&'N' data to 0 & 1

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the input characteristics csv. 
school_info : string
    The filepath to the input school info csv.
output : string
    The filepath to the output annotated characteristics csv. 
no_logging : bool
    Whether to turn off file logging - useful for tests

Outputs
---------
csv file
   annotated characteristics dataset saved at output filepath as a csv file.

"""
import pandas as pd
import argparse

from src.constants import UNKNOWN_CODES, CharacteristicsDataColumns, NA_VALS

# Other code
from src import merge_utils as mu
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import ccis_utils as nu

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input", required=True, 
    type=lambda x: x.strip("'"),
    help="where to find the input characteristics merged data csv"
)
parser.add_argument(
    "--school_info",
    required=True,
    type=lambda x: x.strip("'"),
    help="where to find the input canonicalized secondary school info csv",
)
parser.add_argument(
    "--output", required=True, 
    type=lambda x: x.strip("'"),
    help="where to put the output annotated characteristics data csv"
)
parser.add_argument(
    "--no_file_logging",
    action="store_true",
    help="turn off file logging - useful for tests",
)


def annotated_characteristics_data_validation(df):
    assert not (
        d.isna(df[CharacteristicsDataColumns.year], na_vals=NA_VALS).any()
    ), "Some year values are not filled for the characteristics data. This error likely occured during merging of characteristics datasets."
    assert not (
        d.isna(df[CharacteristicsDataColumns.age], na_vals=NA_VALS).any()
    ), "Some age values are not filled for the characteristics data. Please check all students have birth months and birth years filled in."

def merged_characteristics_data_validation(df):
    pass


if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(
        name=f.get_canonical_filename(__file__),
        debug=args.debug,
        file_logging=(not args.no_file_logging),
    )

    df = d.load_csv(
        args.input,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=False,  # Annotating is nondestructive
        read_as_str=True,
        na_vals=NA_VALS,
        use_na=True,
        logger=logger,
    )
    merged_characteristics_data_validation(df)

    school_df = d.load_csv(
        args.school_info,
        drop_empty=False,
        drop_single_valued=False,
        drop_missing_upns=False,
        drop_duplicates=False,
        read_as_str=True,
        na_vals=NA_VALS,
        use_na=True,
        logger=logger,
    )

    # Merge school info into characteristics data
    logger.info("Merging school info with Characteristics data")
    df = mu.merge_priority_dfs(
        [school_df, df],  # The school df is the higher priority information
        on=CharacteristicsDataColumns.la_establishment_number,
        how="right",
        unknown_vals=UNKNOWN_CODES,
        na_vals=NA_VALS,
    )

    # Split up month/year of birth into two columns
    logger.info("Splitting up birth month and birth year into separate columns")
    df = nu.split_up_birth_month_year(df)

    logger.info(
        "Adding year column for joining datasets that corresponds to the end of school year."
    )
    df[CharacteristicsDataColumns.year] = d.compute_school_end_year(
        df[CharacteristicsDataColumns.data_date]
    )

    logger.info("Compute age of students at the end of prior august")
    df[CharacteristicsDataColumns.age] = nu.compute_age(df)

    y_n_columns = [
        CharacteristicsDataColumns.send_flag,  # Y/N
        CharacteristicsDataColumns.sen_support_flag,
    ]
    logger.info(f"Converting Y/N to 1/0 in columns {y_n_columns}")
    for col in y_n_columns:
        # We must replace with strings because pandas will enforce that the column remain a string type.
        df[col] = df[col].replace({"Y": "1", "N": "0"}).astype(pd.Int8Dtype())

    # Some data validation
    logger.info("Validating annotated data")
    annotated_characteristics_data_validation(df)

    csv_fp = f.tmp_path(args.output, debug=args.debug)

    logger.info(f"Saving annotated data to {csv_fp}")
    df.to_csv(csv_fp, index=False)
