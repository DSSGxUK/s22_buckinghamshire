"""
"""
import argparse
import os
from dataclasses import asdict
from functools import partial
from imblearn.pipeline import Pipeline
import skopt

from sklearn.metrics import (
    f1_score,
    fbeta_score,
    recall_score,
    precision_score,
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score,
)

from src.constants import (
    NA_VALS,
    CensusDataColumns,
    KSDataColumns,
    AttendanceDataColumns,
    BASE_PIPELINE_STEPS,
    UPN,
    YEAR,
    Targets,
)

# Non DVC params but necessary to import
from src.params import (
    get_random_seed,
)

# Other code
from src import cv
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument("--single", action="store_true", help="Use the single upn dataset")
parser.add_argument(
    "--space",
    type=lambda x: x.strip("'"),
    required=True,
    help="which search space to use",
    choices=list(cv.SEARCH_SPACES.keys()),
)
parser.add_argument(
    "--parallel", action="store_true", help="Whether or not to use parallelization"
)
parser.add_argument(
    "--num_thresholds",
    type=int,
    required=False,
    help="Roughly how many thresholds to try to assess. Not passing this will test all the thresholds",
)
parser.add_argument(
    "--input", type=lambda x: x.strip("'"), required=True, help="where to find the input training dataset"
)
parser.add_argument(
    "--target",
    type=lambda x: x.strip("'"),
    required=True,
    choices=list(asdict(Targets).values()),
    help="which target variable to add to csv",
)
parser.add_argument(
    "--checkpoint",
    action="store_true",
    help="whether to save a checkpoint pickle of the hyperparameter search",
)
parser.add_argument(
    "--load_checkpoint",
    action="store_true",
    help="whether to load a saved checkpoint pickle that we can start from",
)
parser.add_argument(
    "--log_results",
    action="store_true",
    help="whether to save the results while optimizing",
)
parser.add_argument(
    "--num_folds",
    default=4,
    type=int,
    help="how many folds to use during k-fold cross validation",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger, logger_getter = l.get_logger(
        name=f.get_canonical_filename(__file__), debug=args.debug, return_getter=True
    )

    logger.info(f'Processing the {"single" if args.single else "multi"} upn dataset')

    df = d.load_csv(
        args.input,
        drop_empty=False,
        drop_single_valued=False,  # Columns here will also be in test set, so don't drop them.
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        use_na=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger,
    )

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

    # Setup scorer
    thresholded_scoring_functions = {
        "f1_binary": partial(f1_score, average="binary"),
        "f2_binary": partial(fbeta_score, beta=2, average="binary"),
        "recall_binary": partial(recall_score, average="binary"),
        "precision_binary": partial(precision_score, average="binary"),
        "matthews_corrcoef": matthews_corrcoef,
        "accuracy": accuracy_score,
    }
    nonthresholded_scoring_functions = {
        "roc_auc_macro": partial(roc_auc_score, average="macro"),
    }
    threshold_scorer = cv.ThresholdingScorer(
        thresholded_scoring_functions=thresholded_scoring_functions,
        num_thresholds=args.num_thresholds,
    )
    nonthreshold_scorer = cv.MultiMetricScorer(
        scoring_functions=nonthresholded_scoring_functions
    )
    scorer = cv.UnionScorer(threshold_scorer, nonthreshold_scorer)

    # Get the input data
    # y should be labels per student and X_indices should be an index of students
    y = df[args.target].astype(int).groupby(level=UPN).max()
    X_indices = y.index
    # The data_df is used by the DataJoinerTransformerFunc to grab the requested indices from the data.
    data_df = d.safe_drop_columns(
        df, columns=asdict(Targets).values(), inplace=True
    ).astype(float)
    df = None  # Clean up unused dataframes

    # Create pipeline
    base_pipeline_steps = BASE_PIPELINE_STEPS
    pipeline = Pipeline(base_pipeline_steps)
    preprocessor = cv.DataJoinerTransformerFunc(data_df)
    postprocessor = cv.identity_postprocessor if args.single else cv.AggregatorTransformerFunc()
    pipeline = cv.PandasEstimatorWrapper(
        pipeline, preprocessor=preprocessor, postprocessor=postprocessor
    )

    # Create the objective function
    _objective = cv.ObjectiveFunction(
        base_model=pipeline,
        y_series=y,
        scorer=scorer,
        watch_score="f2_binary",
        n_folds=args.num_folds,
        random_state=get_random_seed(),
        n_jobs=(-1 if args.parallel else 1),
        results_csv_fp=f.tmp_path(
            f.get_cv_results_filepath(args.space, "single" if args.single else "multi"), debug=args.debug
        )
        if args.log_results
        else None,
        maximize=True,
    )

    # Set up callbacks
    callbacks = []
    if args.checkpoint:
        callbacks.append(
            skopt.callbacks.CheckpointSaver(
                f.tmp_path(f.get_checkpoint_filepath(args.space, "single" if args.single else "multi"), debug=args.debug)
            )
        )
    if args.debug:
        callbacks.append(cv.inspect_optimizer_result)

    # Create the search space
    search_space_dict, n_iter = cv.SEARCH_SPACES[args.space]()
    if not args.single:
        search_space_dict = {
            **search_space_dict,
            "postprocessor__aggregation_method": cv.AGGREGATION_METHODS,
            "postprocessor__aggregation_index": [UPN],
        }
    search_space = cv.search_space_as_list(search_space_dict)
    n_initial_points = min(n_iter, 10)  # 10 is default value of gp_minimize

    # annotate the objective function with a named parameters decorator
    @skopt.utils.use_named_args(search_space)
    def objective(**params):
        return _objective(**params)

    checkpoint_path = f.get_checkpoint_filepath(args.space,  "single" if args.single else "multi")
    if args.load_checkpoint and os.path.exists(checkpoint_path):
        res = skopt.load(args.load_checkpoint)
        x0 = cv.fix_checkpoint_x_iters(res.x_iters, search_space)
        y0 = res.func_vals
        n_iter = n_iter - len(x0)
        n_initial_points = max(0, n_initial_points - len(x0))

    else:
        x0, y0 = None, None

    print(x0, y0)
    opt = skopt.gp_minimize(
        objective,
        search_space,
        n_calls=n_iter,
        n_initial_points=n_initial_points,
        x0=x0,
        y0=y0,
        callback=callbacks,
        verbose=True,
    )
