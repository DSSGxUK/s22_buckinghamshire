"""
Prepares dataset for student predictions

Parameters
-----------
--debug : bool
--single : bool
--train_data : string
    The filepath of where the train data is located. This is to get the column names/features for the model
--unidentified_csv: string
   The filepath of where to save the unidentified csvs
--input : string
    The filepath of where to find the prepared dataset for predictions
--output : string
    The filepath of where to save the feature selected dataset ready for modelling
--student_names : string
    The filepath where csv containing upn and student names is located
--fill_fsme : bool
    Whether to ffill fsme (for multi upn dataset)
Returns
---------
csv file
    feature selected data for predicting on - will have the same columns as the training datasets
csv file 
    csv containing upns and names of unidentifiable students that do not all the data to use for predicting 
    
"""

import pandas as pd
import argparse


from src.constants import UPN, NA_VALS, CensusDataColumns, non_prediction_columns

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import py_utils as py

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--single",
    action="store_true",
    help="should use single upn dataset instead of multi?",
)
parser.add_argument(
    "--input", type=lambda x: x.strip("'"),
    required=True, help="where to find the prediction dataset"
)
parser.add_argument(
    "--output", type=lambda x: x.strip("'"),
    required=True, help="where to save the feature selected dataset"
)
parser.add_argument(
    "--student_names",
    type=lambda x: x.strip("'"),
    required=True,
    help="where to find the dataset with student names",
)
parser.add_argument(
    "--unidentified_csv",
    type=lambda x: x.strip("'"),
    required=True,
    help="where to save the csv with unidentifiable students",
)
parser.add_argument(
    "--train_data", type=lambda x: x.strip("'"),
    required=True, help="where to locate the training dataset"
)
parser.add_argument(
    "--fill_fsme", action="store_true", help="whether to forward fill fsme"
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    logger.info(f'Processing the {"single" if args.single else "multi"} upn dataset')

    train_df = d.load_csv(  # Preserve the training data as is
        args.train_data,
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

    df = d.load_csv(  # Assume we've dropped unnecessary columns and entries
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

    df_additional_data = d.load_csv(  # This holds additional data to annotate unidentified students who were dropped
        args.student_names,
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
        if args.fill_fsme:
            ffil_cols = [CensusDataColumns.fsme_on_census_day]
            logger.info(f"Forward filling fsme column {ffil_cols}.")
            df[ffil_cols] = df[([UPN] + ffil_cols)].groupby(by=UPN).ffill()


    df = d.get_dummies_with_logging(
        df, columns=[CensusDataColumns.fsme_on_census_day], logger=logger
    )
    

    logger.info(
        f'Selecting columns from the {"single" if args.single else "multi"} upn predictions dataset'
    )

    features_for_model = set(train_df.columns) - set(non_prediction_columns)

    logger.info("Filter out any columns not in the train data")
    num_orig_rows = len(df)
    df = df.loc[:, df.columns.isin(features_for_model)]
    logger.info(
        "Add any columns that are in the train data but not in the prediction data"
    )
    for col in features_for_model:
        if col not in df.columns:
            if not d.is_categorical(col):
                logger.warning(
                    f"{col} is not in prediction dataset, but is not categorical. Filling in 0 to suppress error"
                )
            df[col] = 0

    assert set(df.columns) == features_for_model

    nas = df[df.isna().any(axis=1)]
    logger.info(f"Selected rows {len(nas)} that have a missing value")

    df.dropna(inplace=True)  # drop na rows
    logger.info(
        f"Dropped {num_orig_rows - len(df)} rows ({py.safe_divide(num_orig_rows - len(df), num_orig_rows)  * 100}%) because they contained a missing value"
    )

    unidentified_df = pd.DataFrame()
    unidentified_df["upn"] = nas["upn"].unique()

    logger.info(f"Matching student name to unidentified upns")

    add_data = df_additional_data[
        [
            UPN,
            "forename",
            "preferred_surname",
            "establishment_name",
            "nc_year_actual",
            "age",
        ]
    ]

    # add in school name to csv
    unidentified_df = pd.merge(unidentified_df, add_data, how="left", on=UPN)
    # add student_name column
    unidentified_df["student_name"] = unidentified_df["forename"].astype("string")+" "+unidentified_df["preferred_surname"].astype("string")
    #change upn -> UPN column
    unidentified_df = unidentified_df.rename(columns={UPN:"UPN"})


    csv_fp = f.tmp_path(args.output, debug=args.debug)
    unidentified_csv = f.tmp_path(args.unidentified_csv, debug=args.debug)

    logger.info(f"Saving feature selected data to {csv_fp}")
    df.to_csv(csv_fp, index=False)

    logger.info(f"Saving unidentified students data to {unidentified_csv}")
    unidentified_df.to_csv(unidentified_csv, index=False)
