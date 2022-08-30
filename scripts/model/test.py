"""
Predicts on the test dataset and returns metrics of final model
Can recieve multiple models from model_pkls - Determines the final model and final test results from the model that recieves the best test score

Parameters
-----------
--debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.    
--single : bool
    Whether to use the single upn dataset
--input : string
    The filepath of where the input data is located
--output : string
    The filepath of where to save the csv containing the model test scores
--model_output : string
    The filepath of where to save the final model
--model_pkls : pickle
    The filepaths of where the model pickle files are located. Can send multiple filepaths
--target : string
    Which target variable to add to csv

Returns
-----------
csv file
    output dataframe containing threshold and test scores of model
"""

import pandas as pd
import os
import sys
import numpy as np
import argparse
import pickle as pkl

from dataclasses import asdict
from src import data_utils as d
from src import file_utils as f
from src import log_utils as l
from src import cv

from sklearn.metrics import (
    make_scorer,
    f1_score,
    fbeta_score,
    recall_score,
    precision_score,
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score,
)

# should remove this.

# DVC params
from src.constants import (
    CensusDataColumns,
    KSDataColumns,
    AttendanceDataColumns,
    CCISDataColumns,
    UPN,
    NA_VALS,
    Targets,
)


# Non DVC params but necessary to import
from src.params import get_random_seed

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument("--single", action="store_true", help="Use the single upn dataset")
parser.add_argument(
    "--input", type=lambda x: x.strip("'"),required=True, help="where the input test csv is located"
)
parser.add_argument(
    "--output", type=lambda x: x.strip("'"),required=True, help="where to output the evaluation metrics csv"
)
parser.add_argument(
    "--model_output", type=lambda x: x.strip("'"),required=True, help="where to save the final model pkl file"
)
parser.add_argument(
    "--model_pkls",
    type=lambda x: x.strip("'"),required=True,
    nargs="+",
    help="where the model pkl files are located",
)
parser.add_argument(
    "--target",
    type=lambda x: x.strip("'"),required=True,
    choices=list(asdict(Targets).values()),
    help="which target variable to add to csv",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    logger.info(f'Processing the {"single" if args.single else "multi"} upn dataset')

    test_data = args.input
    TEST_RESULTS_CSV_FP = args.output
    MODEL_FP = args.model_pkls
    TARGET = args.target

    # Load the test dataset

    df = d.load_csv(
        test_data,
        drop_empty=True,
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

    # df_original = df.copy(deep=True)
    # breakpoint()
    index_cols = [UPN] if args.single else [UPN, YEAR]
    logger.info(f"Setting index cols {index_cols}")
    df.set_index(index_cols, inplace=True, drop=True)

    extra_cols = [
        CensusDataColumns.has_census_data,
        KSDataColumns.has_ks2_data,
        AttendanceDataColumns.has_attendance_data,
    ]
    logger.info(
        f"Dropping extra cols {extra_cols}, since they are not used for modeling"
    )
    d.safe_drop_columns(df, extra_cols, inplace=True)

    if df.isna().any().any():
        raise NotImplementedError(
            "Data has missing values. We don't know how to handle missing data yet"
        )

    # Get the input data
    # y should be labels per student and X_indices should be an index of students
    y = df[args.target].astype(int).groupby(level=UPN).max()
    X_indices = y.index
    # The data_df is used by the DataJoinerTransformerFunc to grab the requested indices from the data.
    data_df = d.safe_drop_columns(
        df, columns=asdict(Targets).values(), inplace=True
    ).astype(float)
    df = None  # Clean up unused dataframes

    # Load the model.pkl files
    model_test_results = []
    fbetas = []
    models_dict = []

    for FP in MODEL_FP:
        with open(FP, "rb") as model_file:
            model = pkl.load(model_file)

        # breakpoint()
        # Run on test data
        logger.info(
            f'Predicting using {model["estimator"]}, threshold_type: {model["threshold_type"]}, threshold: {model["threshold"]}'
        )

        # Create pipeline
        # PIPELINE_STEPS =
        pipeline = model["estimator"]
        # breakpoint()
        # pipeline = Pipeline(PIPELINE_STEPS)
        preprocessor = cv.DataJoinerTransformerFunc(data_df)
        postprocessor = (
            cv.identity_postprocessor if args.single else cv.AggregatorTransformerFunc()
        )
        pipeline = cv.PandasEstimatorWrapper(
            pipeline,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            threshold=model["threshold"],
            threshold_type=model["threshold_type"],
        )

        # breakpoint()
        # Output probablities and predictions based on threshold
        if pipeline.threshold_type == "decision_function":
            y_prob = pipeline.decision_function(X_indices)
            predictions = [0 if i <= pipeline.threshold else 1 for i in y_prob]
        elif pipeline.threshold_type == "predict_proba":
            y_prob = pipeline.predict_proba(X_indices)
            predictions = [0 if i <= pipeline.threshold else 1 for i in y_prob]
        else:
            raise ValueError(f"Unknown threshold type {pipeline.threshold_type}")

        # breakpoint()

        # Get scores
        f1 = f1_score(y, predictions, average="binary")
        fbeta = fbeta_score(y, predictions, beta=2, average="binary")
        recall = recall_score(y, predictions, average="binary")
        precision = precision_score(y, predictions, average="binary")
        mcc = matthews_corrcoef(y, predictions)
        accuracy = accuracy_score(y, predictions)
        roc_auc = roc_auc_score(y, y_prob, average="macro")  # send y_prob

        # breakpoint()

        test_results = {
            "threshold_type": pipeline.threshold_type,
            "threshold": pipeline.threshold,
            "f1": f1,
            "fbeta": fbeta,
            "recall": recall,
            "precision": precision,
            "mcc": mcc,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        }

        # Save test results
        logger.info(f"Fbeta2 score for final model on test dataset: {fbeta}")

        fbetas.append(fbeta)
        model_test_results.append(test_results)
        model_dict = {
            "estimator": pipeline.estimator,
            "threshold": pipeline.threshold,
            "threshold_type": pipeline.threshold_type,
        }
        models_dict.append(model_dict)
        # breakpoint()
    # breakpoint()

    final_model = models_dict[np.argmax(fbetas)]
    final_model_test_results = model_test_results[np.argmax(fbetas)]

    # breakpoint()
    FINAL_MODEL_FP = args.model_output
    FINAL_MODEL_FP = f.tmp_path(args.model_output,debug=args.debug)
    TEST_RESULTS_CSV_FP = f.tmp_path(TEST_RESULTS_CSV_FP,debug=args.debug)
        

    logger.info(f"Saving final model to pickle file {FINAL_MODEL_FP}")
    pkl.dump(final_model, open(FINAL_MODEL_FP, "wb"))

    logger.info(f"Saving test results to {TEST_RESULTS_CSV_FP}")
    pd.DataFrame({k: [v] for k, v in final_model_test_results.items()}).to_csv(
        TEST_RESULTS_CSV_FP, index=False
    )

    # breakpoint()
#    else:
#        logger.error("We don't have code yet to test the multi-upn dataset")
#        raise NotImplementedError()
