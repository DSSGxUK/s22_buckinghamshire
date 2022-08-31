"""
This stage selects the features to use for modeling. We mainly
use this to clean out missing values because our models do not know
how to handle them.

Parameters
----------
--debug: bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
--single: bool
    Passed if the input file is a single UPN dataset
--input: string (filepath)
    Filepath of the input csv file. This is a required parameter.
--output: string (filepath)
    Filepath of where to save the output csv file. This is a required parameter.
--forward_fill_fsme_column: bool
    If passed as an argument, forward fill in missing free school meal values for student-years based on last known value for that student.     This is only valid for the multi-upn dataset.
--feature_selection_method: string
    Which feature selection method to use. This is a required parameter.
--remove_mostly_missing_threshold: float (between 0 and 1)
    Threshold used for the `remove_mostly_missing` feature selection method. Columns will be rejected if their percentage of non-missing         values falls below this threshold.

Returns
-------
csv file
   csv file saved at `--output` filepath ready to be used for modelling, with missing values removed.  
    
"""

import argparse
from dataclasses import asdict


from src.constants import FeatureSelectionMethods, UPN, NA_VALS, CensusDataColumns

# Other code
from src.error_utils import error_with_logging
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d


def remove_mostly_missing(df, threshold, na_vals, logger=l.PrintLogger):
    """
    This is the `remove_mostly_missing` feature selection method. It
    will remove columns that have fewer than `threshold` fraction of non-
    missing values. Finally, it will remove any rows (students if `--single` is used,
    student-years otherwise) that have any remaining missing values after removing
    those columns. A properly chosen `threshold` should keep the number of students
    removed down to a minimum. For example, threshold=0.7 removes most ks2 columns
    and only 28 remaining students with missing values in their rows on the single-upn
    dataset.

    Parameters
    ----------
    df: pd.DataFrame
        This is the dataframe holding the data to apply feature selection to.
        threshold: This is the threshold used to remove columns. If a column has
        less than a threshold fraction of non-missig values, it is dropped.
    na_vals: iterable (e.g. list, set)
        These are the values in the data considered as missing. Usually
        these are ['', '#VALUE!', pd.NA, np.nan].
    logger: logging.logger (or something that implements the same logging methods)
        This is the python logger to use for logging progress. By default,
        this is the `PrintLogger` defined in `log_utils`.

    Returns
    -------
    pd.DataFrame
        The input `df` after the `remove_mostly_missing` method of feature
        selection has been applied.
    """
    df = df.copy()

    na_percents = d.isna(df, na_vals=na_vals).sum(axis=0) / len(df)
    keep_cols = na_percents.index[na_percents < (1 - threshold)]
    drop_cols = na_percents.index[na_percents >= (1 - threshold)]

    #    breakpoint()

    logger.info(
        f"Dropping columns {list(drop_cols)} because they had less than {args.remove_mostly_missing_threshold} non-missing values"
    )
    df = df.loc[:, keep_cols]

    #    breakpoint()

    num_orig_rows = len(df)
    df.dropna(inplace=True)  # drop na rows

    logger.info(
        f"Dropped {num_orig_rows - len(df)} rows ({(num_orig_rows - len(df)) / num_orig_rows  * 100}%) because they contained a missing value"
    )
    # breakpoint()

    # breakpoint()

    return df


parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--single",
    action="store_true",
    help="should use single upn dataset instead of multi?",
)
parser.add_argument(
    "--input", type=lambda x: x.strip("'"), required=True, help="where to find the categorical dataset"
)
parser.add_argument(
    "--output", type=lambda x: x.strip("'"), required=True, help="where to put dataset after feature selection"
)
parser.add_argument(
    "--forward_fill_fsme_column",
    action="store_true",
    help="if we are running on the multi-upn dataset, should we fill in missing fsme values with the most recent known value for that student?",
)
parser.add_argument(
    "--feature_selection_method",
    type=lambda x: x.strip("'"), required=True,
    choices=list(asdict(FeatureSelectionMethods).values()),
    help="which feature selection method to use",
)
parser.add_argument(
    "--remove_mostly_missing_threshold",
    required=False,
    type=float,
    help="if using the 'remove_mostly_missing' feature selection method, what threshold to use? \
                        Columns will be rejected if their percentage of non-missing values falls below the threshold. ",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    logger.info(f'Processing the {"single" if args.single else "multi"} upn dataset')

    if args.single:
        if args.forward_fill_fsme_column:
            error_with_logging(
                f"'forward_fill_fsme_columns' cannot be passed with 'single'. It is only valid for processing the multi-upn dataset",
                logger,
                ValueError,
            )
        fill_fsme_col = False
    else:
        fill_fsme_col = args.forward_fill_fsme_column

    df = d.load_csv(  # Modeling dataset so we drop unnecessary columns and entries
        args.input,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        use_na=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger,
    )

    if not args.single:
        if fill_fsme_col:
            ffil_cols = [CensusDataColumns.fsme_on_census_day]
            logger.info(f"Forward filling fsme column {ffil_cols}.")
            df[ffil_cols] = df[([UPN] + ffil_cols)].groupby(by=UPN).ffill()

    df = d.get_dummies_with_logging(
        df, columns=[CensusDataColumns.fsme_on_census_day], logger=logger
    )

    # breakpoint()

    if args.feature_selection_method == FeatureSelectionMethods.none:
        if args.remove_mostly_missing_threshold is not None:
            error_with_logging(
                f"You cannot pass a value for 'remove_mostly_missing_threshold' if 'feature_selection_method' is none. You passed {args.remove_mostly_missing_threshold}.",
                logger,
                ValueError,
            )
        pass
    elif args.feature_selection_method == FeatureSelectionMethods.remove_mostly_missing:
        if args.remove_mostly_missing_threshold is None:
            error_with_logging(
                f"You must pass a value for 'remove_mostly_missing_threshold' if 'feature_selection_method' is 'remove_mostly_missing'. You passed {args.remove_mostly_missing_threshold}.",
                logger,
                ValueError,
            )
        df = remove_mostly_missing(
            df,
            threshold=args.remove_mostly_missing_threshold,
            na_vals=NA_VALS,
            logger=logger,
        )
    else:
        error_with_logging(
            f"{args.feature_selection_method} is not a valid feature selection method. It must be one of {list(asdict(FeatureSelectionMethods).values())}. If it is, then your method is currently not implemented.",
            logger,
            ValueError,
        )
    # breakpoint()
    # breakpoint()

    csv_fp = args.output
    if args.debug:
        csv_fp = f.tmp_path(csv_fp)

    logger.info(f"Saving categorical data to {csv_fp}")
    df.to_csv(csv_fp, index=False)
