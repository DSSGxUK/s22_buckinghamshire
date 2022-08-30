"""
This stage is a *premerge* stage. This means it takes its input data
and prepares it for the `merge_multi_upn` stage that creates
a modeling dataset. In this case, the data is the CCIS data (aka NEET data).

This stage will perform 
- a drop of
- only keep columns that will not be used for modeling unless they are useful information for later stages. Dropped include
  -  columns that cannot be used for modeling because they are only available after the student is of age >= 15. (e.g. `intended_yr11_destination` or `activity_code`).
  - columns that are categorical but too numerous or too fine-grained.
  - date based columns that the model cannot take as input.
- some type conversion of columns that should be numeric.
- aggregation by UPN.

Parameters
----------
input : str
    The filepath to the input CCIS annotated csv. This is usually
    `../data/interim/neet_annotated.csv`.
output : str
    The filepath to the output CCIS premerge csv. This is usually
    `../data/interim/neet_premerge.csv`

Other Parameters
----------------
N/A

Validation
------
...

"""

import argparse

# DVC Params
from src.constants import (
    NEET_PREMERGE_AGG,
    CCISDataColumns,
    NEET_PREMERGE_AGG,
    NA_VALS,
)

# Other code
from src import data_utils as d
from src import file_utils as f
from src import log_utils as l
from src import aggregation_utils as au


def annotated_neet_data_validation(df):
    """
    TODO: Any validation for incoming data can be done here.
    You should write your validation in terms of `assert`
    statements or raise an exception if any check fails.
    """
    pass


parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input", required=True, help="where to find the input CCIS annotated csv"
)
parser.add_argument(
    "--output", required=True, help="where to put the output CCIS premerge csv"
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    df = d.load_csv(  # We are now going to filter out the necessary columns
        args.input,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=True,  # duplicate rows are treated as one entry and duplicates are dropped
        read_as_str=True,  # This will ensure values are not cast to floats
        use_na=True,  # and this will ensure empty values are read as nan
        drop_missing_upns=True,  # Now we drop the empty UPNs
        na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger,
    )

    logger.info(f"Initial row count {len(df)}")
    logger.info(f"Initial column count {len(df.columns)}")

    # Do some validation
    logger.info("Validating incoming CCIS annotated data")
    annotated_neet_data_validation(df)

    # Can't use age because it won't make sense when we aggregate
    modeling_columns = [
        CCISDataColumns.upn,
        CCISDataColumns.ethnicity,
        CCISDataColumns.gender,
        CCISDataColumns.sen_support_flag,
        CCISDataColumns.send_flag,
        CCISDataColumns.birth_month,
        CCISDataColumns.characteristic_code,
        CCISDataColumns.level_of_need_code,
    ]
    target_columns = [
        CCISDataColumns.neet_ever,
        CCISDataColumns.unknown_ever,
    ]
    useful_columns = [
        CCISDataColumns.unknown_currently,
        CCISDataColumns.compulsory_school_always,
        CCISDataColumns.birth_year,
    ]
    logger.info(
        "Keeping only columns that can be passed to model trained for pre-age15 prediction and other useful columns."
    )
    logger.info(
        f"These columns are {modeling_columns + target_columns + useful_columns}"
    )
    df = df[modeling_columns + target_columns + useful_columns]

    # Create categorical column for characteristic codes and level_of_need
    df = d.get_dummies_with_logging(
        df,
        columns=[
            CCISDataColumns.characteristic_code,
            CCISDataColumns.level_of_need_code,
            CCISDataColumns.send_flag,
            CCISDataColumns.sen_support_flag,
        ],
        logger=logger,
    )

    # Aggregate by upn
    aggregation_dict = au.build_agg_dict(
        NEET_PREMERGE_AGG, columns=df.columns, logger=logger
    )
    df = au.gby_agg_with_logging(
        df, groupby_column=CCISDataColumns.upn, agg_dict=aggregation_dict, logger=logger
    )
    logger.info(f"{len(df)} rows after aggregation. This is the number of unique upns")

    logger.info(f"Final row count {len(df)}")
    logger.info(f"Final column count {len(df.columns)}")

    csv_fp = f.tmp_path(args.output, debug=args.debug)
    logger.info(f"Saving categorical data to {csv_fp}")
    df.to_csv(csv_fp, index=False)
