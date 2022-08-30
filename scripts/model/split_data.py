"""
Split data into train test datasets for modelling

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string
    The filepath of where the input csv is located
split : float
    The percentage of upns to hold out in the test set
train_output : string
    The filepath of where to save the train csv output file
test_output : string
    The filepath of where to save the test csv output file
single : bool
    Whether the single upn dataset is passed 
target : string
    which target variable to add to the csv

Returns
---------
csv file
    train dataset saved at the `train_output` filepath 
csv file
    test dataset saved at the `test_output` filepath 

"""

from dataclasses import asdict
import argparse
from sklearn.model_selection import train_test_split

from src import data_utils as d
from src import file_utils as f
from src import log_utils as l
from src import merge_utils as mu

from src.constants import UPN, Targets


parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument("--single", action="store_true", help="Use the single upn dataset")
parser.add_argument(
    "--split", type=float, help="The percentage of upns to hold out in the test set"
)
parser.add_argument("--input", type=lambda x: x.strip("'"),required=True, help="where the input csv is located")
parser.add_argument(
    "--train_output", type=lambda x: x.strip("'"),required=True, help="where the output train csv is located"
)
parser.add_argument(
    "--test_output", type=lambda x: x.strip("'"),required=True, help="where the output test csv is located"
)
parser.add_argument(
    "--target",
    type=lambda x: x.strip("'"),required=True,
    choices=list(asdict(Targets).values()),
    help="which target variable to add to csv",
)
parser.add_argument(
    "--seed",
    default=1,
    type=int,
    help="seed for random number generator when splitting data into train and test sets",
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
        file_logging=(not args.no_file_logging),
    )

    logger.info(f'Processing the {"single" if args.single else "multi"} upn dataset')

    # Load the dataset

    df = d.load_csv(args.input, convert_dtypes=True, logger=logger)

    # Split data into train and test [stratify on target column]

    test_output_fp = f.tmp_path(args.test_output, debug=args.debug)
    train_output_fp = f.tmp_path(args.train_output, debug=args.debug)
    if args.single:
        logger.info(
            f"Performing train test split with test_split {args.split} stratified on {args.target} on the single upn dataset"
        )
        train_single_df, test_single_df = train_test_split(
            df,
            test_size=args.split,
            random_state=args.seed,
            shuffle=True,
            stratify=df[args.target],
        )

        logger.info(
            f"Writing training data to {train_output_fp} and writing test data to {test_output_fp}"
        )
        train_single_df.to_csv(train_output_fp, index=False)
        test_single_df.to_csv(test_output_fp, index=False)

    else:
        multi_upns = df.groupby(UPN, as_index=False)[args.target].max()

        logger.info(
            f"Performing train test split with test_split {args.split} stratified on {args.target} on multi-upn dataset"
        )
        train_multi_df, test_multi_df = train_test_split(
            multi_upns,
            test_size=args.split,
            random_state=args.seed,
            shuffle=True,
            stratify=multi_upns[args.target],
        )
        logger.info(f"Train dataset has {len(train_multi_df)} upns.")
        logger.info(f"Test dataset has {len(test_multi_df)} upns.")

        logger.info(f"Merging split upns back into multi upn data")
        train_multi_df = mu.merge_priority_data(df, train_multi_df, how="right", on=UPN)
        test_multi_df = mu.merge_priority_data(df, test_multi_df, how="right", on=UPN)
        logger.info(f"Train dataset has {len(train_multi_df)} rows.")
        logger.info(f"Test dataset has {len(test_multi_df)} rows.")

        logger.info(
            f"Writing training data to {train_output_fp} and writing test data to {test_output_fp}"
        )
        train_multi_df.to_csv(train_output_fp, index=False)
        test_multi_df.to_csv(test_output_fp, index=False)
