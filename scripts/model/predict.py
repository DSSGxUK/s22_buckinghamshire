"""
Predicts on the current student cohort and outputs a csv file with risk scores, neet predictions, features & shapley values for each student to be imported to the power bi dahsboard   

Parameters
-----------
--debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.    
--single : bool
    Whether to use the single upn dataset
--input : string
    The filepath of where the input data is located (this is data on students in the current cohort to predict on)
--output : string
    The filepath of where to save the csv containing the predictions and risk scores for each student 
--model_pkl : pickle
    The filepath of where the model pickle file is located
--roni_threshold : string
    The filepath of where the csv with the roni tool threshold is located
--additional_data : string
    The filepath of where the additional data is located that needs to be added back to the final output for the power bi dashboard 

Returns
-----------
csv file
    input dataframe with additional columns for `predictions`, `probabilities`, scaled probabilites and roni scores for importing into power bi dashboard
"""

import argparse
import pandas as pd
import pickle as pkl
import shap
from dataclasses import asdict

# need to remove this
from src import cv
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import roni
from src import merge_utils as mu

from src.constants import (
    CensusDataColumns,
    KSDataColumns,
    AttendanceDataColumns,
    CharacteristicsDataColumns,
    UPN,
    YEAR,
    NA_VALS,
    UNKNOWN_CODES,
    Targets,
)


def rescale_range(data, old_low, old_high, new_low, new_high):
    """
    Function to scale the probabilities for the power bi dashboard.
    The probabilities above and below the threshold are scaled separately.

    Parameters
    -----------
    data : pd.Series
        the data to rescale
    old_low : float
        the low value of the column to scale to match new_low
    old_high : float
        the high value of the column to scale to match new_high
    new_low : float
        the new low value to scale old_low to match.
    new_high : float
        the new high value to scale old_high to match.

    Returns
    -----------
    pd.Series
        The data after rescaling
    """
    slope = (new_high - new_low) / (old_high - old_low)
    return slope * (data - old_low) + new_low


parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--single", action="store_true", help="whether to use the single upn dataset"
)
parser.add_argument(
    "--output",
    type=lambda x: x.strip("'"),
    required=True,
    help="where to output the csv file containing predictions and data for power bi dashboard",
)
parser.add_argument(
    "--input",
    type=lambda x: x.strip("'"),
    required=True,
    help="where the input csv containing students for prediciting on is located",
)
parser.add_argument(
    "--model_pkl",
    type=lambda x: x.strip("'"), required=True, help="where the model pkl file is located"
)
parser.add_argument(
    "--roni_threshold",
    type=lambda x: x.strip("'"),
    required=True,
    help="where the csv with roni tool threshold is located",
)
parser.add_argument(
    "--additional_data",
    type=lambda x: x.strip("'"),
    required=True,
    help="where the csv with additional data is located",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if not args.single:
        raise NotImplementedError("This code is not yet complete for hanlding the multi-UPN data.")

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    logger.info(f'Processing the {"single" if args.single else "multi"} upn dataset')

    MODEL_FP = args.model_pkl

    # Load the dataset for prediction

    df = d.load_csv(
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

    # The df is used by the DataJoinerTransformerFunc to grab the requested indices from the data.
    d.safe_drop_columns(
        df, columns=asdict(Targets).values(), inplace=True
    )


    # Load the model.pkl file
    with open(MODEL_FP, "rb") as model_file:
        model = pkl.load(model_file)

    logger.info(
        f'Predicting using {model["estimator"]}, threshold_type: {model["threshold_type"]}, threshold: {model["threshold"]}'
    )

    # Create pipeline
    preprocessor = cv.DataJoinerTransformerFunc(df)
    postprocessor = (
        cv.identity_postprocessor if args.single else cv.AggregatorTransformerFunc()
    )
    pipeline = cv.PandasEstimatorWrapper(
        model["estimator"],
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        threshold=model["threshold"],
        threshold_type=model["threshold_type"],
    )
    # Output probablities and predictions based on threshold

    X_indices = df.index.get_level_values(UPN).unique()
    if pipeline.threshold_type == "decision_function":
        y_prob = pipeline.decision_function(X_indices)
        lower_bound = y_prob.min()
        upper_bound = y_prob.max()
    elif pipeline.threshold_type == "predict_proba":
        y_prob = pipeline.predict_proba(X_indices)
        lower_bound = 0.
        upper_bound = 1.
    else:
        raise ValueError(f"Unknown threshold type {pipeline.threshold_type}")
    predictions = (y_prob >= pipeline.threshold).astype(pd.Int8Dtype())


    # Save predictions & probabilities
    PREDICTIONS_CSV_FP = args.output
    output_predictions = pd.DataFrame(index=X_indices, columns=[])
    output_predictions["predictions"] = predictions
    output_predictions["probabilities"] = y_prob  # "probabilities" is a bad name since this may not actually be probabilities if decision_function is used

    # this section adds a new column with the probabilities scaled between 0 and 10, with a threshold of 7, for the power bi dashboard
    logger.info(f"Scaling probabilities for power bi dashboard")
    new_threshold = 7  # for power bi dashboard
    new_upper_bound = 10
    new_lower_bound = 0
    threshold = pipeline.threshold

    output_predictions["neet_scanr"] = d.empty_series(len(output_predictions), index=output_predictions.index)
    # For the neets rescale their scores to fit in between new_threshold and new_upper_bound
    output_predictions.loc[output_predictions["predictions"] == 1, "neet_scanr"] = rescale_range(
        output_predictions.loc[output_predictions["predictions"] == 1, "probabilities"],
        old_low=threshold,
        old_high=upper_bound,
        new_low=new_threshold,
        new_high=new_upper_bound
    )
    # For the eets rescale their scores to fit in between new_lower_bound and new_threshold
    output_predictions.loc[output_predictions["predictions"] == 0, "neet_scanr"] = rescale_range(
        output_predictions.loc[output_predictions["predictions"] == 0, "probabilities"],
        old_low=lower_bound,
        old_high=threshold,
        new_low=new_lower_bound,
        new_high=new_threshold
    )

    feature_names = df.columns
    # Feature importance
    logger.info(f"Calculating feature importances")

    try:
        model_coefs = pd.Series(pipeline.estimator[-1].coef_[0], index=feature_names)
    except AttributeError:
        model_coefs = pd.Series(
            pipeline.estimator[-1].feature_importances_, index=feature_names
        )
    # Order the features in order of most important to least
    feature_names = model_coefs.sort_values(ascending=False, key=abs).index
    #breakpoint()
    
    # Shapley values - might not work with multi-upn
    logger.info(f"Calculating shapley values")
    if len(df) == 0:
        logger.warning(f"No data in input dataframe.")    
    explainer = shap.Explainer(pipeline.estimator[-1])
    # explainer = shap.LinearExplainer(pipeline.estimator[-1], masker=shap.maskers.Impute(data=X), data=X)
    shap_df = pipeline.shapley_values(X_indices, explainer=explainer)
    shap_df = shap_df.loc[:, feature_names]  # Reorder featurs in orde of most to least important
    shap_df.rename(columns=lambda s: "shap_" + s, inplace=True)  # Prepend with shap, so we know which ones are shapley values
    
    
    # calculate roni tool scores
    logger.info(f"Calculating roni tool scores for each student")
    # get roni tool threhsold
    RONI_FP = args.roni_threshold
    roni_results = pd.read_csv(RONI_FP)
    roni_threshold = roni_results["threshold"][0]
    roni_df = roni.calculate_roni_scores(df, threshold=roni_threshold)
    
    final_output = pd.concat(
        [output_predictions, shap_df, roni_df], axis=1
    )  # put together the final output

    # Merge with the latest data for the student
    latest_data = df.groupby(level=UPN).last()
    final_output = mu.merge_priority_data(final_output, latest_data, how="left", on=UPN, unknown_vals=UNKNOWN_CODES, na_vals=NA_VALS)

    # att <85%
    final_output["att_less_than_85"] = roni_df["roni_att_below_85"]
    # level of need column
    final_output["level_of_need"] = d.empty_series(len(final_output), index=final_output.index)    
    if d.to_categorical(CharacteristicsDataColumns.level_of_need_code, "1") in final_output.columns :
        final_output.loc[final_output[d.to_categorical(CharacteristicsDataColumns.level_of_need_code, "1")] == 1, "level_of_need"] = "intensive support"
    if d.to_categorical(CharacteristicsDataColumns.level_of_need_code, "2") in final_output.columns :        
        final_output.loc[final_output[d.to_categorical(CharacteristicsDataColumns.level_of_need_code, "2")]== 1, "level_of_need"] = "supported"
    if d.to_categorical(CharacteristicsDataColumns.level_of_need_code, "3") in final_output.columns :        
        final_output.loc[final_output[d.to_categorical(CharacteristicsDataColumns.level_of_need_code, "3")]== 1, "level_of_need"] = "minimum intervention"
        
    # send_flag column
    final_output["send_flag"] = d.empty_series(len(final_output), index=final_output.index)   
    if (d.to_categorical(CharacteristicsDataColumns.send_flag, "0")) and (d.to_categorical(CharacteristicsDataColumns.send_flag, "1")) in final_output.columns :        
        final_output.loc[(final_output[d.to_categorical(CharacteristicsDataColumns.send_flag, "0")]==0)&(final_output[d.to_categorical(CharacteristicsDataColumns.send_flag, "1")]==0), "send_flag"] = pd.NA
        final_output.loc[(final_output[d.to_categorical(CharacteristicsDataColumns.send_flag, "0")]==0)&(final_output[d.to_categorical(CharacteristicsDataColumns.send_flag, "1")]==1), "send_flag"] = 1
        final_output.loc[(final_output[d.to_categorical(CharacteristicsDataColumns.send_flag, "0")]==1)&(final_output[d.to_categorical(CharacteristicsDataColumns.send_flag, "1")]==0), "send_flag"] = 0
    
    logger.info(f"Adding data back in from previous datasets")
    # here can load data for adding back in previous data.
    pre_model_df = d.load_csv(
        args.additional_data,
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
    pre_model_df.set_index(UPN, inplace=True, drop=True)
    
    # adding extra columns for power bi
    # add student_name column
    pre_model_df["student_name"] = pre_model_df["forename"].astype("string")+" "+pre_model_df["preferred_surname"].astype("string")
    # school name
    pre_model_df["school_name"] = pre_model_df["establishment_name"]
    # sch_yr
    pre_model_df["sch_yr"] = pre_model_df["nc_year_actual"]
    # school postcode
    pre_model_df["school_postcode"] = pre_model_df["establishment_postcode"]

    #merge additional data to final output
    final_output = mu.merge_priority_data(final_output, pre_model_df, how="left", on=UPN, unknown_vals=UNKNOWN_CODES, na_vals=NA_VALS)
    #edit excluded_authorised column names
    final_output = final_output.rename(columns={"excluded_authorised":"excluded_authorised_percent1"})
    final_output = final_output.rename(columns={"excluded_authorised_exact":"excluded_authorised"})

    csv_fp = f.tmp_path(PREDICTIONS_CSV_FP, debug=args.debug)
    logger.info(f"Saving predictions to {csv_fp}")
    final_output.to_csv(csv_fp, index=True)  # Index is True since the index is UPN
