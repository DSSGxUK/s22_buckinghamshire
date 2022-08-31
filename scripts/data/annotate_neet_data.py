"""
Annotates NEET CCIS dataset
TODO: Test

1. Add school establishment info to the datasets merging on la_establishment_number
2. Add a column for NEET and for Unknown
3. Split up month and year of birth into separate columns
4. Add year column for joining datasets
5. Add column with student age 
6. Propagate NEET and unknowns to other instances of same upn
7. Convert columns with 'Y'&'N' data to 0 & 1

Rewrite codes to true names in new column. E.g. create a column 'activity_name' that has the name corresponding to the code in 'activity_code'. TODO 

Validated in 0019.0-am-debug-annotate-neet.ipynb

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath to the input ccis-neet csv. 
school_info : string
    The filepath to the input school info csv.
output : string
    The filepath to the output annotated ccis-neet csv. 
no_logging : bool
    Whether to turn off file logging - useful for tests

Returns
---------
csv file
   annotated neet dataset saved at output filepath as a csv file.

"""
import pandas as pd
import argparse

from src.constants import UNKNOWN_CODES, CCISDataColumns, NA_VALS

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
    help="where to find the input CCIS merged data csv"
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
    help="where to put the output annotated CCIS data csv"
)
parser.add_argument(
    "--no_file_logging",
    action="store_true",
    help="turn off file logging - useful for tests",
)


def annotated_neet_data_validation(df):
    assert not (
        d.isna(df[CCISDataColumns.year], na_vals=NA_VALS).any()
    ), "Some year values are not filled for the CCIS data. This error likely occured during merging of CCIS datasets."
    assert not (
        d.isna(df[CCISDataColumns.age], na_vals=NA_VALS).any()
    ), "Some age values are not filled for the CCIS data. Please check all students have birth months and birth years filled in."
    # TODO add validation for neet_ever and unknown_ever columns


def merged_neet_data_validation(df):
    assert not (
        d.isna(df[CCISDataColumns.young_persons_id]).any()
    ), f"Some young_persons_id are missing for the CCIS data. Please check your source data and check to make sure none of the young_persons_id cells have any of the values {NA_VALS}."


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
    merged_neet_data_validation(df)
    df[CCISDataColumns.ccis_period_end] = pd.to_datetime(
        df[CCISDataColumns.data_date], errors="raise"
    )  # This will set the data date specific column for CCIS data

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

    # Merge school info into neet data
    logger.info("Merging school info with NEET data")
    df = mu.merge_priority_dfs(
        [school_df, df],  # The school df is the higher priority information
        on=CCISDataColumns.la_establishment_number,
        how="right",
        unknown_vals=UNKNOWN_CODES,
        na_vals=NA_VALS,
    )
    # Add column for neet and unknown
    logger.info("Adding columns for neet and unknown status")
    df = d.add_column(
        df, CCISDataColumns.neet, nu.neet(df).astype(pd.Int8Dtype())
    )  # These are booleans so use Int8 to save space.
    df = d.add_column(
        df, CCISDataColumns.unknown, nu.unknown(df).astype(pd.Int8Dtype())
    )
    df = d.add_column(
        df,
        CCISDataColumns.compulsory_school,
        nu.in_compulsory_school(df).astype(pd.Int8Dtype()),
    )

    # Split up month/year of birth into two columns
    logger.info("Splitting up birth month and birth year into separate columns")
    df = nu.split_up_birth_month_year(df)

    logger.info(
        "Adding year column for joining datasets that corresponds to the end of school year."
    )
    df[CCISDataColumns.year] = d.compute_school_end_year(
        df[CCISDataColumns.ccis_period_end]
    )

    logger.info("Compute age of students at the end of prior august")
    df[CCISDataColumns.age] = nu.compute_age(df)

    # Propogate neet and unknown statuses to other instances of the ypid
    logger.info("Marking NEET and unknown students")
    # We use ypid because UPN may be missing. YPID is assumed to all be non-missing, and this is validated upon loading the merged df.
    # Alternatively, we could use .all() instead of .max().
    neet_propogated_df = (
        df[
            [
                CCISDataColumns.young_persons_id,
                CCISDataColumns.neet,
                CCISDataColumns.unknown,
            ]
        ]
        .groupby(by=CCISDataColumns.young_persons_id)
        .max()
        .reset_index()
    )
    df = df.merge(
        neet_propogated_df,
        on=CCISDataColumns.young_persons_id,
        how="left",
        suffixes=(None, "_ever"),
    )

    logger.info("Marking students whose current (in latest ccis dataset) is unknown")
    latest_date = df[CCISDataColumns.ccis_period_end].max()
    # It is possible the student has multiple rows, with one of which they are unknown
    # and the others where they are known. However, these rows can have different
    # activity dates and currency lapsed dates. The reasonable thing to do would be
    # to classify a student as currently unknown if all of their last known statuses in the
    # latest dataset have currency lapsed. However, this is somewhat complicated logic
    # and to keep things simple, we just mark a student as currently unknown if
    # one of the entries for the student in the latest dataset is an unknown activity.
    # Alternatively, we could use .all() instead of .max().
    unknown_currently_df = (
        df.loc[
            df[CCISDataColumns.ccis_period_end] == latest_date,
            [CCISDataColumns.young_persons_id, CCISDataColumns.unknown],
        ]
        .groupby(by=CCISDataColumns.young_persons_id)
        .max()
        .reset_index()
    )
    df = df.merge(
        unknown_currently_df,
        on=CCISDataColumns.young_persons_id,
        how="left",
        suffixes=(None, "_currently"),
    )
    df[CCISDataColumns.unknown_currently] = df[
        CCISDataColumns.unknown_currently
    ].fillna(
        0
    )  # Anyone who is NA is not in the latest dataset and is thus not currently unknown

    logger.info("Marking students who have no data outside of compulsory school")
    # We take the min so that if a student only has data from compulsory school, then that column will always be
    # 1 and the output would be 1. Alternatively, we could use .any().
    compulsory_school_always_df = (
        df[[CCISDataColumns.young_persons_id, CCISDataColumns.compulsory_school]]
        .groupby(by=CCISDataColumns.young_persons_id)
        .min()
        .reset_index()
    )
    df = df.merge(
        compulsory_school_always_df,
        on=CCISDataColumns.young_persons_id,
        how="left",
        suffixes=(None, "_always"),
    )

    # NEET data
    y_n_columns = [
        CCISDataColumns.send_flag,  # Y/N
        CCISDataColumns.currency_lapsed,  # Y/N
        CCISDataColumns.youth_contract_indicator,
        CCISDataColumns.sen_support_flag,
    ]
    logger.info(f"Converting Y/N to 1/0 in columns {y_n_columns}")
    for col in y_n_columns:
        # We must replace with strings because pandas will enforce that the column remain a string type.
        df[col] = df[col].replace({"Y": "1", "N": "0"}).astype(pd.Int8Dtype())

    # Some data validation
    logger.info("Validating annotated data")
    annotated_neet_data_validation(df)

    csv_fp = f.tmp_path(args.output, debug=args.debug)

    logger.info(f"Saving annotated data to {csv_fp}")
    df.to_csv(csv_fp, index=False)
