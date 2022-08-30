"""
Creates single UPN dataset from multipe UPN dataset.

This step creates a new dataset from the multiple UPN dataset.
It takes the multiple UPN dataset (with multiple years of data per student)
and aggregates data across the years into a single row per each UPN. 
The aggregation method for each column is specified in `MULTI_UPN_CATEGORICAL_TO_SINGLE_AGGS` 

Parameters
----------
--debug: bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
--input: string (filepath)
    Filepath of the input csv file. This is a required parameter.
--output: string (filepath)
    Filepath of where to save the output csv file. This is a required parameter. 

Returns
-------
csv file
   single upn dataset saved at output filepath as a csv file.  

"""

import argparse

from src.constants import NA_VALS, MULTI_UPN_CATEGORICAL_TO_SINGLE_AGGS, UPN

from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import aggregation_utils as au

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input", required=True, help="where to find multi upn categorical dataset"
)
parser.add_argument(
    "--output", required=True, help="where to put single upn categorical dataset"
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    # loading the multiple UPN dataset file
    df = d.load_csv(
        args.input,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=False,
        na_vals=NA_VALS,
        logger=logger,
        convert_dtypes=True,
        downcast=True,
        use_na=True,
    )
    # applying aggregation function
    agg_dict = au.build_agg_dict(
        MULTI_UPN_CATEGORICAL_TO_SINGLE_AGGS, columns=df.columns
    )
    df = au.gby_agg_with_logging(
        df, groupby_column=UPN, agg_dict=agg_dict, logger=logger
    )

    csv_fp = f.tmp_path(args.output, debug=args.debug)

    logger.info(f"Saving aggregated(Single UPN) data to {csv_fp}")
    df.to_csv(csv_fp, index=False)
