"""
Keeps the KS2 column data from the KS4 dataset. We do this as we will not have KS4 exam data at the time of predicting whether students are NEET (before Year 11). 

Parameters
----------
--debug: bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
--input: string (filepath)
    Filepath of the KS4 csv file. This is a required parameter.
--output: string (filepath)
    Filepath of where to save the output csv file. This is a required parameter. 

Returns
-------
csv file
    ks2 data ready to be merged with other datasets
"""

import argparse
from dataclasses import asdict

# DVC Params
from src.constants import (
    KS2OriginalColumns,
    SchoolInfoColumns,
    UPN,
    NA_VALS,
    UNKNOWN_CODES
)

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import merge_utils as mu

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--ks4_input", type=lambda x: x.strip("'"),
    required=True, help="where to find input csv file: annotated ks4 data"
)
parser.add_argument(
    "--ks2_input", type=lambda x: x.strip("'"),
    required=True, help="where to find input csv file: annotated ks2 data"
)
parser.add_argument(
    "--output", type=lambda x: x.strip("'"),
    required=True, help="where to output the produced csv file"
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    ks4_df = d.load_csv(  # Modeling dataset so we drop unnecessary columns and entries
        args.ks4_input,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,  # This will ensure values are not cast to floats
        use_na=True,  # and this will ensure empty values are read as nan
        drop_missing_upns=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        logger=logger,
    )

    logger.info(f"Initial KS4 row count {len(ks4_df)}")
    logger.info(f"Initial KS4 column count {len(ks4_df.columns)}")

    ks2_df = d.load_csv(  # Modeling dataset so we drop unnecessary columns and entries
        args.ks2_input,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=True,  # This will ensure values are not cast to floats
        use_na=True,  # and this will ensure empty values are read as nan
        drop_missing_upns=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        logger=logger,
    )

    logger.info(f"Initial KS2 row count {len(ks2_df)}")
    logger.info(f"Initial KS2 column count {len(ks2_df.columns)}")

    logger.info(f"Merging KS2 and KS4 data")
    df = mu.merge_priority_data(ks4_df, ks2_df, on=UPN, how="outer", unknown_vals=UNKNOWN_CODES, na_vals=NA_VALS)

    keep_cols = list(
        set(asdict(KS2OriginalColumns).values()) | {
        SchoolInfoColumns.establishment_type,
        SchoolInfoColumns.establishment_status,
    })
    df = df[keep_cols]

    logger.info(f"Final row count {len(df)}")
    logger.info(f"Final column count {len(df.columns)}")

    csv_fp = f.tmp_path(args.output, debug=args.debug)

    logger.info(f"Saving ks2 filtered data to {csv_fp}")
    df.to_csv(csv_fp, index=False)
