"""
Calculates RONI scores for each student on the train dataset to obtain the threshold with the best score & then run on the test dataset using the best threshold to output evaluation metrics.

Parameters
-----------
--debug : bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory. 
--train_data : string
    The filepath where the train dataset is located
--test_data : string
    The filepath where the test dataset is located
--output_test_metrics : string
    The filepath where the test metrics csv will be saved
--output_roni_scores : string
    The filepath where the roni scores csv will be saved
--target : string
    Which target variable to add to csv

Returns
-----------
csv file
    csv file containing evaluation metrics of roni tool performance

"""

import argparse
import pandas as pd
import numpy as np
from dataclasses import asdict

from sklearn.metrics import f1_score, fbeta_score, recall_score, precision_score

# DVC filepaths
from src.constants import UPN, NA_VALS, Targets

from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import roni

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--train_data", required=True, help="where the train data csv is located"
)
parser.add_argument(
    "--test_data", required=True, help="where the test data csv is located"
)
parser.add_argument(
    "--output_test_metrics",
    required=True,
    help="where to save the evaluation metrics csv",
)
parser.add_argument(
    "--target",
    required=True,
    choices=list(asdict(Targets).values()),
    help="which target variable to add to csv",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    logger.info(f"Running roni tool calculation")
    data_fp = args.train_data
    TARGET = args.target

    df = d.load_csv(
        data_fp,
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

    df = df.copy()

    if df.isna().any().any():
        logger.error(
            "Data has missing values. We don't know how to handle missing data yet"
        )
        raise NotImplementedError()

    roni_df = roni.calculate_roni_scores(df)

    thresholds = range(1, max(roni_df["roni_score"]) + 1)
    print(thresholds)
    fbetas = []

    fscores = []
    for th in thresholds:
        roni_df["y_pred_" + str(th)] = np.where((roni_df["roni_score"] >= th), 1, 0)
        # breakpoint()

        print("Threshold", th)
        print(
            "recall_score :",
            recall_score(df[TARGET].astype("int64"), roni_df["y_pred_" + str(th)]),
        )
        print(
            "precision_score",
            precision_score(df[TARGET].astype("int64"), roni_df["y_pred_" + str(th)]),
        )
        print(
            "f1_score",
            f1_score(df[TARGET].astype("int64"), roni_df["y_pred_" + str(th)]),
        )
        print(
            "fbeta_score",
            fbeta_score(
                df[TARGET].astype("int64"), roni_df["y_pred_" + str(th)], beta=2
            ),
        )

        fscores.append(
            f1_score(df[TARGET].astype("int64"), roni_df["y_pred_" + str(th)])
        )
        fbetas.append(
            fbeta_score(
                df[TARGET].astype("int64"), roni_df["y_pred_" + str(th)], beta=2
            )
        )

    print("Best Fbeta threshold:", np.argmax(fbetas) + 1)
    print("Best Fscore threshold:", np.argmax(fscores) + 1)

    best_threshold = np.argmax(fbetas) + 1

    """Run RONI tool calculation on test dataset"""

    data_fp = args.test_data

    df = d.load_csv(
        data_fp,
        drop_empty=True,
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        use_na=True,
        # upn_col=UPN,
        # na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger,
    )

    df = df.copy()

    if df.isna().any().any():
        logger.error(
            "Data has missing values. We don't know how to handle missing data yet"
        )
        raise NotImplementedError()

    roni_df = roni.calculate_roni_scores(df, threshold=best_threshold)

    recall = recall_score(df[TARGET].astype("int64"), roni_df["roni_prediction"])
    precision = precision_score(df[TARGET].astype("int64"), roni_df["roni_prediction"])
    f1 = f1_score(df[TARGET].astype("int64"), roni_df["roni_prediction"])
    fbeta = fbeta_score(df[TARGET].astype("int64"), roni_df["roni_prediction"], beta=2)

    print("Threshold:", best_threshold)
    print("Test scores:")
    print("recall_score :", recall)
    print("precision_score", precision)
    print("f1_score", f1)
    print("fbeta_score", fbeta)

    scores = [best_threshold, recall, precision, f1, fbeta]
    scores_df = pd.DataFrame(
        [np.transpose(scores)],
        columns=["threshold", "recall", "precision", "f1", "fbeta"],
    )

    # breakpoint()

    if args.debug:
        RONI_RESULTS = f.tmp_path(args.output_test_metrics)
    RONI_RESULTS = args.output_test_metrics

    scores_df.to_csv(RONI_RESULTS, index=False)
