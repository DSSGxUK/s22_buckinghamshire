"""
TODO: Docs
TODO: Test
"""
import argparse
import pandas as pd

from src import data_utils as d
from src import merge_utils as mu
from src import log_utils as l
from src import file_utils as f

from src.constants import UPN, CCISDataColumns, KSDataColumns


parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--ccis_input",
    required=True,
    help="where to find the CCIS (NEET dataset) annotated csv",
)
parser.add_argument(
    "--ks_input", required=True, help="where to find the Key Stage annotated csv"
)
parser.add_argument(
    "--output", required=True, help="where to put the produced CCIS - Key Stage EDA csv"
)
parser.add_argument(
    "--no_file_logging",
    action="store_true",
    help="turn off file logging - useful for tests",
)

if __name__ == "__main__":

    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(
        name=f.get_canonical_filename(__file__),
        debug=args.debug,
        file_logging=args.no_file_logging,
    )

    neet_df = d.load_csv(args.ccis_input, convert_dtypes=True, logger=logger)

    ks_df = d.load_csv(  # Modeling dataset so we drop unnecessary columns and entries
        args.ks_input, convert_dtypes=True, logger=logger
    )

    ccis_filter = [
        CCISDataColumns.upn,
        CCISDataColumns.neet_ever,
        CCISDataColumns.unknown_ever,
        CCISDataColumns.compulsory_school,
    ]
    logger.info(f"Filtering out {ccis_filter} columns from ccis annotated dataset.")
    neet_df = neet_df[ccis_filter]
    logger.info(f"Dropping rows for student in compulsory school")
    # This ensures that when we aggregate, the students remaining are those with data after compulsory school
    neet_df = neet_df.loc[
        ~(neet_df[CCISDataColumns.compulsory_school].astype(bool))
    ].drop(CCISDataColumns.compulsory_school, axis=1)
    logger.info(
        f"Grouping the ccis data by UPN to get a dataframe of students mapped to their post-16 statuses"
    )
    neet_df = neet_df.groupby(by=CCISDataColumns.upn).max().reset_index()

    ks_filter = [
        KSDataColumns.upn,
        KSDataColumns.deprivation_indicator_idaci_score,
        KSDataColumns.attainment_8_score,
        KSDataColumns.ks2_prior_band,
    ]
    logger.info(
        f"Filtering out {ks_filter} columns from the key stage annotated dataset."
    )
    ks_df = ks_df[ks_filter]

    logger.info(f"Merging the ccis data with key stage data")
    neet_ks_df = mu.merge_priority_dfs([neet_df, ks_df], on=UPN, how="inner")

    csv_fp = f.tmp_path(args.output, debug=args.debug)

    logger.info(f"Saving ccis - key-stage eda data to {csv_fp}")
    neet_ks_df.to_csv(csv_fp, index=False)
