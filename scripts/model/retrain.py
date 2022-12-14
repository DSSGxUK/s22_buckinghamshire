"""
Re-train best model(s) from cross validation on full training dataset

Parameters
-----------
--debug : bool
--single : bool
    Whether to use the single upn dataset
--input : string
    The filepath of where to find the training dataset
--model_metrics : string
    The filepath of where to find the csv containing the models, parameters and metrics from the cross-validation
--target : string
    Which target variable to add to csv
--model_output_best : string
    The filepath of where to save the model pickle file (model with best threshold)
--model_output_mean : string
    The filepath of where to save the model pickle file (model with mean threshold)

Returns
----------
pickle file
    pickle file contianing final model
    
"""

import argparse
from dataclasses import asdict
import os
import pandas as pd
import pickle as pkl
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from lightgbm.sklearn import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE

from src.constants import (
    NA_VALS,
    CensusDataColumns,
    KSDataColumns,
    AttendanceDataColumns,
    UPN,
    YEAR,
    Targets,
    PipelineSteps,
)

# Non DVC params but necessary to import

# Other code
from src import cv
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument("--single", action="store_true", help="Use the single upn dataset")
parser.add_argument(
    "--input", type=lambda x: x.strip("'"), required=True, help="where to find the input training dataset"
)
parser.add_argument(
    "--model_metrics", nargs="+",
    type=lambda x: x.strip("'"), required=True,
    help="where to find the csv containing the best model metrics",
)
parser.add_argument(
    "--target",
    required=True,
    type=lambda x: x.strip("'"), choices=list(asdict(Targets).values()),
    help="which target variable to add to csv",
)
parser.add_argument(
    "--model_output_best",
    type=lambda x: x.strip("'"), required=True,
    help="where to save the pickle of the best thresholded model",
)
parser.add_argument(
    "--model_output_mean",
    type=lambda x: x.strip("'"), required=True,
    help="where to save the pickle of the mean thresholded model",
)
parser.add_argument("--num_of_models",type=int, required=False, help = "how many of the top models to retrain and test")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger, logger_getter = l.get_logger(
        name=f.get_canonical_filename(__file__), debug=args.debug, return_getter=True
    )

    logger.info(f'Processing the {"single" if args.single else "multi"} upn dataset')

    # Load train dataset
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

    # Get the input data
    # y should be labels per student and X_indices should be an index of students
    # breakpoint()

    # breakpoint()
    y = df[args.target].astype(int).groupby(level=UPN).max()
    X_indices = y.index
    # The data_df is used by the DataJoinerTransformerFunc to grab the requested indices from the data.
    data_df = d.safe_drop_columns(
        df, columns=asdict(Targets).values(), inplace=True
    ).astype(float)
    df = None  # Clean up unused dataframes

    logger.info(
        f"Loading model metrics from {args.model_metrics} to select best model from cv"
    )
    
    if args.num_of_models :
        num_of_models = args.num_of_models
    else :
        num_of_models = 1
        
    # Load csv with best model parameters and threshold
    all_metrics = [pd.read_csv(metrics_path) for metrics_path in args.model_metrics if os.path.exists(metrics_path)]
    if len(all_metrics) == 0:
        raise ValueError(f"None of the metrics csvs {args.model_metrics} existed. Please run hyperparameter search.")
    #best_metrics = max(all_metrics, key=lambda m: m["f2_binary_mean"].max())

    #best_model = best_metrics.loc[
    #    best_metrics["f2_binary_mean"].idxmax()]  # get best model parameters
    top_models = (pd.concat(all_metrics)).sort_values(by="f2_binary_mean",ascending=False).reset_index(drop=True)[:num_of_models]
    #top_models = [m.sort_values(by="f2_binary_mean",ascending=False)[:5] for m in all_metrics] #get top n models
    #top_models = pd.concat(top_models).reset_index(drop=True)

    #best_threshold = best_model["f2_binary__threshold_best"]
    #best_threshold = best_model["f2_binary__threshold_mean"]
    # best_model = best_model.iloc[:(best_model.index.get_loc('postprocessor__aggregation_index'))+1]

    # loop through all the top models
    
    for idx,best_model in top_models.iterrows() :
        best_threshold = best_model["f2_binary__threshold_best"]
        mean_threshold = best_model["f2_binary__threshold_mean"]
        
        model_est = best_model.estimator__model.split("(")[0]
        model_oversamp = best_model.estimator__oversampling.split("(")[0]
        model_scaler = best_model.estimator__scaler.split("(")[0]
        model_imputation = best_model.estimator__imputation.split("(")[0]

        params = {
            k: v if v != "None" else None
            for k, v in best_model.to_dict().items()
            if k.startswith("estimator")
        }
        if "estimator" in params:
            params.pop("estimator")
        if "estimator__steps" in params:
            params.pop("estimator__steps")

        # get hyperparams for each pipeline step
        oversamp_params = best_model.loc[
            best_model.index.str.startswith("estimator__oversampling__")
        ]
        estimator_params = best_model.loc[
            best_model.index.str.startswith("estimator__model__")
        ]
        imputation_params = best_model.loc[
            best_model.index.str.startswith("estimator__imputation__")
        ]
        scaler_params = best_model.loc[
            best_model.index.str.startswith("estimator__scaler__")
        ]
        # breakpoint()
        oversamp_params.index = oversamp_params.index.str.replace(
            "estimator__oversampling__", ""
        )
        estimator_params.index = estimator_params.index.str.replace(
            "estimator__model__", ""
        )
        imputation_params.index = imputation_params.index.str.replace(
            "estimator__imputation__", ""
        )
        scaler_params.index = scaler_params.index.str.replace("estimator__scaler__", "")

        constructor = {
            "RandomOverSampler": lambda: RandomOverSampler(),
            "SMOTE": lambda: SMOTE(),
            "LGBMClassifier": lambda: LGBMClassifier(),
            "None": lambda: None,
            "KNeighborsClassifier": lambda: KNeighborsClassifier(),
            "LinearSVC": lambda: LinearSVC(),
            "SVC": lambda: SVC(),
            "LogisticRegression": lambda: LogisticRegression(),
            "StandardScaler": lambda: StandardScaler(),
        }

        params["estimator__oversampling"] = constructor[model_oversamp]()
        params["estimator__model"] = constructor[model_est]()
        params["estimator__scaler"] = constructor[model_scaler]()
        params["estimator__imputation"] = constructor[model_imputation]()

        from pprint import pprint

        pprint(params)
        # breakpoint()

        logger.info(f"Creating pipeline from best model in {args.model_metrics}")

        # get threshold_type
        if model_est == "LGBMClassifier":
            threshold_type = "predict_proba"
        else:
            threshold_type = "decision_function"
        # else :
        #    raise ValueError(f'Unknown threshold type for {estimator}')

        # Create pipeline
        PIPELINE_STEPS = [
            (PipelineSteps.oversampling, None),
            (PipelineSteps.imputation, None),
            (PipelineSteps.scaler, None),
            (PipelineSteps.model, None),
        ]

        pipeline = Pipeline(PIPELINE_STEPS)
        preprocessor = cv.DataJoinerTransformerFunc(data_df)
        postprocessor = (
            cv.identity_postprocessor if args.single else cv.AggregatorTransformerFunc()
        )
        pipeline_best_thresh = cv.PandasEstimatorWrapper(
            pipeline,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            threshold=best_threshold,
            threshold_type=threshold_type,
        )
        pipeline_mean_thresh = cv.PandasEstimatorWrapper(
            pipeline,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            threshold=mean_threshold,
            threshold_type=threshold_type,
        )

        #breakpoint()
        if params["estimator__oversampling"] :
            params["estimator__oversampling__random_state"] = int(params["estimator__oversampling__random_state"])
        else :   
            for k in list(params.keys()):
                if k.startswith('estimator__oversampling__'):
                    del params[k]
            #breakpoint()
        
        #if not np.isnan(params["estimator__oversampling__random_state"]):
        #breakpoint()
        #breakpoint()
        pipeline_best_thresh.set_params(**params)
        pipeline_mean_thresh.set_params(**params)

        # breakpoint()

        logger.info(f"Training model on whole training dataset {args.input}")

        # Train model with mean and best threshold
        # breakpoint()
        pipeline_best_thresh.fit(X_indices, y)
        pipeline_mean_thresh.fit(X_indices, y)

        model_best_thresh = {
            "estimator": pipeline_best_thresh.estimator,
            "threshold": pipeline_best_thresh.threshold,
            "threshold_type": pipeline_best_thresh.threshold_type,
        }
        model_mean_thresh = {
            "estimator": pipeline_mean_thresh.estimator,
            "threshold": pipeline_mean_thresh.threshold,
            "threshold_type": pipeline_mean_thresh.threshold_type,
        }

        #breakpoint()

        BEST_MODEL_FP = args.model_output_best 
        MEAN_MODEL_FP = args.model_output_mean

        if num_of_models > 1 :
            substr = ".pkl"
            str_idx = BEST_MODEL_FP.index(substr)
            BEST_MODEL_FP = BEST_MODEL_FP[:str_idx] + "_"+str(idx) + BEST_MODEL_FP[str_idx:]
            str_idx = MEAN_MODEL_FP.index(substr)
            MEAN_MODEL_FP = MEAN_MODEL_FP[:str_idx] + "_"+str(idx) + MEAN_MODEL_FP[str_idx:]

        BEST_MODEL_FP = f.tmp_path(BEST_MODEL_FP, debug=args.debug)
        MEAN_MODEL_FP = f.tmp_path(MEAN_MODEL_FP, debug=args.debug)

        logger.info(f"Saving best thresholded model to pickle file {BEST_MODEL_FP}")
        logger.info(f"Saving mean thresholded model to pickle file {MEAN_MODEL_FP}")
        
        #breakpoint()

        pkl.dump(model_best_thresh, open(BEST_MODEL_FP, "wb"))
        pkl.dump(model_mean_thresh, open(MEAN_MODEL_FP, "wb"))

        # testing how to load the model
        # with open(BEST_MODEL_FP, 'rb') as model_file:
        #    model = pkl.load(model_file)

        # breakpoint()
