"""
This file contains the filepaths of all the raw, interim and processed datasets that are used in the data pipeline as well as filepaths of csv outputs from the model
"""

from glob import glob
import os

def _read_savedate(fp):
    basename = os.path.basename(fp)
    filename = os.path.splitext(basename)[0]
    return filename.split("_")[-1]

DATA_DIR = "../data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = "../models"
FINAL_MODELS_DIR = os.path.join(MODELS_DIR, "final")
INTERIM_MODELS_DIR = os.path.join(MODELS_DIR, "interim")
METRICS_DIR = "../metrics"
PLOTS_DIR = "../plots"
LOGS_DIR = "../logs"
TMP_DIR = "../tmp"
PARAMS_DIR = "../src/params"
RESULTS_DIR = "../results"
INTERIM_RESULTS_DIR = os.path.join(RESULTS_DIR, "interim")

DVC_YAML = "./dvc.yaml"
PARAMS_YAML = "./params.yaml"

MULTI_UPN_FP = os.path.join(INTERIM_DIR, f"multi_upn.csv")
MULTI_UPN_BASIC_FP = os.path.join(INTERIM_DIR, f"multi_upn_basic.csv")
MULTI_UPN_UNKNOWNS_FP = os.path.join(INTERIM_DIR, f"multi_upn_unknowns.csv")
MULTI_UPN_PREDICTION_FP = os.path.join(INTERIM_DIR, f"multi_upn_prediction.csv")

MULTI_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"multi_upn_categorical.csv"
)
MULTI_UPN_CATEGORICAL_BASIC_FP = os.path.join(
    PROCESSED_DATA_DIR, f"multi_upn_basic_categorical.csv"
)

SINGLE_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"single_upn_categorical.csv"
)
SINGLE_UPN_CATEGORICAL_BASIC_FP = os.path.join(
    PROCESSED_DATA_DIR, f"single_upn_basic_categorical.csv"
)
FEATURE_SELECTED_MULTI_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"feature_selected_multi_upn_categorical.csv"
)
FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"feature_selected_single_upn_categorical.csv"
)
FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_BASIC_FP = os.path.join(
    PROCESSED_DATA_DIR, f"feature_selected_single_upn_basic_categorical.csv"
)
UNKNOWNS_CSV_FP_SINGLE = os.path.join(METRICS_DIR, "unknown_features.csv")


# EDA datasets
NEET_KS_EDA_FP = os.path.join(PROCESSED_DATA_DIR, "neet_ks_eda.csv")

# RONI
RONI_RESULTS_FP = os.path.join(METRICS_DIR, "roni_test_results.csv")
RONI_SCORES_FP = os.path.join(METRICS_DIR, "roni_scores.csv")

# Separate datasets for pre and post covid years

PRE_COVID_MULTI_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"pre_covid_multi_upn_categorical.csv"
)
POST_COVID_MULTI_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"post_covid_multi_upn_categorical.csv"
)
FS_PRE_COVID_MULTI_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"fs_pre_covid_multi_upn_categorical.csv"
)
FS_POST_COVID_MULTI_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"fs_post_covid_multi_upn_categorical.csv"
)
PRE_COVID_SINGLE_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"pre_covid_single_upn_categorical.csv"
)
POST_COVID_SINGLE_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"post_covid_single_upn_categorical.csv"
)
FS_PRE_COVID_SINGLE_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"fs_pre_covid_single_upn_categorical.csv"
)
FS_POST_COVID_SINGLE_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"fs_post_covid_single_upn_categorical.csv"
)

# Train Test Datasets for SingleUPN
SINGLE_TRAIN_FP = os.path.join(PROCESSED_DATA_DIR, f"train_singleUPN.csv")
SINGLE_TEST_FP = os.path.join(PROCESSED_DATA_DIR, f"test_singleUPN.csv")
SINGLE_TRAIN_BASIC_FP = os.path.join(PROCESSED_DATA_DIR, f"train_singleUPN_basic.csv")
SINGLE_TEST_BASIC_FP = os.path.join(PROCESSED_DATA_DIR, f"test_singleUPN_basic.csv")

# Train Test Datasets for MultiUPN
MULTI_TRAIN_FP = os.path.join(PROCESSED_DATA_DIR, f"train_multiUPN.csv")
MULTI_TEST_FP = os.path.join(PROCESSED_DATA_DIR, f"test_multiUPN.csv")

# Model tranining outputs
MODEL_FP = os.path.join(FINAL_MODELS_DIR, "model.pkl")
PIPELINE_MODEL_FP = os.path.join(INTERIM_MODELS_DIR, "pipeline_model.pkl")
BASE_MODEL_FP = os.path.join(INTERIM_MODELS_DIR, "base_model.pkl")
CHECKPOINT_FP = os.path.join(INTERIM_MODELS_DIR, "checkpoint.pkl")
CV_METRICS_FP = os.path.join(METRICS_DIR, "cv.yaml")
CV_RESULTS_CSV_FP = os.path.join(METRICS_DIR, "cv.csv")

# Training and test data
TRAIN_FP = os.path.join(PROCESSED_DATA_DIR, "train.csv")

# Model tranining outputs
MODEL_FP = os.path.join(FINAL_MODELS_DIR, "model.pkl")
PIPELINE_MODEL_FP = os.path.join(INTERIM_MODELS_DIR, "pipeline_model.pkl")
BASE_MODEL_FP = os.path.join(INTERIM_MODELS_DIR, "base_model.pkl")
CV_METRICS_FP = os.path.join(METRICS_DIR, "cv.yaml")
CV_RESULTS_CSV_FP = os.path.join(METRICS_DIR, "cv.csv")
TEST_RESULTS_CSV_FP = os.path.join(METRICS_DIR, "test_results.csv")

RONI_TEST_RESULTS = os.path.join(METRICS_DIR, "roni_test_results.csv")
RONI_TEST_RESULTS_BASIC = os.path.join(METRICS_DIR, "roni_test_results_basic.csv") 

LGBM1_METRICS_MULTI = os.path.join(METRICS_DIR, "lgbm1_multi.csv")
LGBM1_METRICS_SINGLE = os.path.join(METRICS_DIR, "lgbm1_single.csv")
LGBM2_METRICS_SINGLE = os.path.join(METRICS_DIR, "lgbm2_single.csv")
LGBM1_SINGLE_CHECKPOINT_FP = os.path.join(INTERIM_MODELS_DIR, "lgbm1_single.pkl")
LGBM2_SINGLE_CHECKPOINT_FP = os.path.join(INTERIM_MODELS_DIR, "lgbm2_single.pkl")
LGBM1_METRICS_SINGLE_BASIC = os.path.join(METRICS_DIR, "lgbm1_basic_single.csv")
LGBM1_SINGLE_CHECKPOINT_BASIC_FP = os.path.join(INTERIM_MODELS_DIR, "lgbm1_single_basic.pkl")

MODEL_BEST_THRESH_MULTI = os.path.join(
    INTERIM_MODELS_DIR, "model_best_thresh_multi.pkl"
)
MODEL_MEAN_THRESH_SMULTI = os.path.join(
    INTERIM_MODELS_DIR, "model_mean_thresh_multi.pkl"
)
FINAL_MODEL_MULTI_FP = os.path.join(FINAL_MODELS_DIR, "model_multi.pkl")
TEST_RESULTS_MULTI_CSV_FP = os.path.join(METRICS_DIR, "multi_test_results.csv")

MODEL_BEST_THRESH_SINGLE = os.path.join(
    INTERIM_MODELS_DIR, "model_best_thresh_single.pkl"
)
MODEL_MEAN_THRESH_SINGLE = os.path.join(
    INTERIM_MODELS_DIR, "model_mean_thresh_single.pkl"
)
MODEL_BEST_THRESH_SINGLE_BASIC = os.path.join(
    INTERIM_MODELS_DIR, "model_best_thresh_single_basic.pkl"
)
MODEL_MEAN_THRESH_SINGLE_BASIC = os.path.join(
    INTERIM_MODELS_DIR, "model_mean_thresh_single_basic.pkl"
)
FINAL_MODEL_SINGLE_FP = os.path.join(FINAL_MODELS_DIR, "model_single.pkl")
FINAL_MODEL_SINGLE_BASIC_FP = os.path.join(FINAL_MODELS_DIR, "model_single_basic.pkl")

TEST_RESULTS_SINGLE_CSV_FP = os.path.join(METRICS_DIR, "single_test_results.csv")
TEST_RESULTS_SINGLE_BASIC_CSV_FP = os.path.join(METRICS_DIR, "single_test_results_basic.csv")
# Prediction & unknown datasets
SINGLE_UPN_CATEGORICAL_PREDICT_FP = os.path.join(
    PROCESSED_DATA_DIR, "predict_singleUPN.csv"
)
SINGLE_UPN_CATEGORICAL_PREDICT_BASIC_FP = os.path.join(
    PROCESSED_DATA_DIR, "predict_singleUPN_basic.csv"
)
SINGLE_UPN_CATEGORICAL_UNKNOWNS_FP = os.path.join(
    PROCESSED_DATA_DIR, "unknowns_singleUPN.csv"
)
FS_SINGLE_UPN_CATEGORICAL_PREDICT_FP = os.path.join(
    PROCESSED_DATA_DIR, "fs_predict_singleUPN.csv"
)
FS_SINGLE_UPN_CATEGORICAL_PREDICT_BASIC_FP = os.path.join(
    PROCESSED_DATA_DIR, "fs_predict_singleUPN_basic.csv"
)
FS_SINGLE_UPN_CATEGORICAL_UNKNOWNS_FP = os.path.join(
    PROCESSED_DATA_DIR, "fs_unknowns_singleUPN.csv"
)
MULTI_UPN_CATEGORICAL_PREDICT_FP = os.path.join(
    PROCESSED_DATA_DIR, "predict_multiUPN.csv"
)
MULTI_UPN_CATEGORICAL_UNKNOWNS_FP = os.path.join(
    PROCESSED_DATA_DIR, "unknowns_multiUPN.csv"
)
FS_MULTI_UPN_CATEGORICAL_PREDICT_FP = os.path.join(
    PROCESSED_DATA_DIR, "fs_predict_multiUPN.csv"
)
FS_MULTI_UPN_CATEGORICAL_UNKNOWNS_FP = os.path.join(
    PROCESSED_DATA_DIR, "fs_unknowns_multiUPN.csv"
)

# Predictions outputs
PREDICTIONS_CSV_FP_SINGLE = os.path.join(RESULTS_DIR, "predictions.csv")
UNKNOWN_PREDICTIONS_CSV_FP_SINGLE = os.path.join(RESULTS_DIR, "unknown_predictions.csv")
PREDICTIONS_CSV_FP_MULTI = os.path.join(RESULTS_DIR, "predictions_multi.csv")
UNKNOWN_PREDICTIONS_CSV_FP_MULTI = os.path.join(
    RESULTS_DIR, "unknown_predictions_multi.csv"
)
PREDICTIONS_CSV_FP_SINGLE_BASIC = os.path.join(INTERIM_RESULTS_DIR, "predictions_basic.csv")
PREDICTIONS_CSV_FP_SINGLE_ORIG = os.path.join(INTERIM_RESULTS_DIR, "predictions_orig.csv") 
# Unidentified students datasets
SINGLE_UNIDENTIFIED_PRED_FP = os.path.join(
    RESULTS_DIR, "unidentified_students_single.csv"
)
SINGLE_UNIDENTIFIED_PRED_BASIC_FP = os.path.join(
    INTERIM_RESULTS_DIR, "unidentified_students_single_basic.csv"
)
SINGLE_UNIDENTIFIED_PRED_ORIG_FP = os.path.join(
    INTERIM_RESULTS_DIR, "unidentified_students_single_orig.csv"
)
MULTI_UNIDENTIFIED_PRED_FP = os.path.join(
    RESULTS_DIR, "unidentified_students_multi.csv"
)
SINGLE_UNIDENTIFIED_UNKS_FP = os.path.join(
    RESULTS_DIR, "unidentified_unknowns_single.csv"
)
MULTI_UNIDENTIFIED_UNKS_FP = os.path.join(
    RESULTS_DIR, "unidentified_unknowns_multi.csv"
)

# additional data to add to predictions for power bi dashboard

ADDITIONAL_DATA_FP = os.path.join(PROCESSED_DATA_DIR, "additional_data.csv")

# School data Paths
SCHOOL_INFO_CSV_FP = os.path.join(RAW_DATA_DIR, "secondary_schools_original.csv")
SCHOOL_INFO_CANONICALIZED_CSV_FP = os.path.join(INTERIM_DIR, "secondary_schools.csv")

# School Census Paths
SCHOOL_CENSUS_CSV_DIR = os.path.join(RAW_DATA_DIR, "census_original_csv")
SCHOOL_CENSUS_CSV_FPS = {
    _read_savedate(fp): fp for fp in glob(os.path.join(SCHOOL_CENSUS_CSV_DIR, "*.csv"))
}

SCHOOL_CENSUS_CANONICALIZED_CSV_DIR = os.path.join(
    INTERIM_DIR, "census_canonicalized_csv"
)
SCHOOL_CENSUS_CANONICALIZED_CSV_FPS = {

    date: os.path.join(SCHOOL_CENSUS_CANONICALIZED_CSV_DIR, f"census_{date}.csv")
    for date in SCHOOL_CENSUS_CSV_FPS.keys()

}
SCHOOL_CENSUS_MERGED_FP = os.path.join(INTERIM_DIR, "census_merged.csv")
SCHOOL_CENSUS_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, "census_annotated.csv")
SCHOOL_CENSUS_PREMERGE_CSV_FP = os.path.join(INTERIM_DIR, "census_premerge.csv")

# Attendance filepaths
ATTENDANCE_CSV_DIR = os.path.join(RAW_DATA_DIR, "attendance_original_csv")
ATTENDANCE_DATA_CSV_FPS = {
    _read_savedate(fp): fp for fp in glob(os.path.join(ATTENDANCE_CSV_DIR, "*.csv"))
}

# from os import listdir
# att_raw_csvs = os.listdir(ATTENDANCE_CSV_DIR)
# for f in att_raw_csvs :

# ATTENDANCE_DATA_CSV_FPS = {}

ATTENDANCE_CANONICALIZED_CSV_DIR = os.path.join(
    INTERIM_DIR, "attendance_canonicalized_csv"
)
ATTENDANCE_CANONICALIZED_CSV_FPS = {

    date: os.path.join(ATTENDANCE_CANONICALIZED_CSV_DIR, f"attendance_{date}.csv")
    for date in ATTENDANCE_DATA_CSV_FPS.keys()

}

ATTENDANCE_MERGED_FP = os.path.join(INTERIM_DIR, "attendance_merged.csv")
ATTENDANCE_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f"attendance_annotated.csv")
ATTENDANCE_EXACT_CSV_FP = os.path.join(INTERIM_DIR, f"attendance_exact.csv")
ATTENDANCE_PERCENT1_CSV_FP = os.path.join(INTERIM_DIR, f"attendance_percent1.csv")
ATTENDANCE_PERCENT2_CSV_FP = os.path.join(INTERIM_DIR, f"attendance_percent2.csv")
ATTENDANCE_NORMED_CSV_FP = os.path.join(INTERIM_DIR, f"attendance_normed.csv")

# KS4 Files - we were not given an excel file so it does not show up here.
# If you have an excel file, either convert it to CSV first, or break it into
# individual sheets and run it through our tool.
KS4_CSV_DIR = os.path.join(RAW_DATA_DIR, "ks4_original_csv")
KS4_DATA_CSV_FPS = {
    _read_savedate(fp): fp for fp in glob(os.path.join(KS4_CSV_DIR, "*.csv"))
}
KS4_CANONICALIZED_CSV_DIR = os.path.join(INTERIM_DIR, "ks4_canonicalized_csv")
KS4_CANONICALIZED_CSV_FPS = {

    date: os.path.join(KS4_CANONICALIZED_CSV_DIR, f"ks4_{date}.csv")
    for date in KS4_DATA_CSV_FPS.keys()

}
KS4_MERGED_FP = os.path.join(INTERIM_DIR, "ks4_merged.csv")
KS4_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f"ks4_annotated.csv")

# KS2 Files
KS2_CSV_DIR = os.path.join(RAW_DATA_DIR, "ks2_original_csv")
KS2_CSV_FPS = {
    _read_savedate(fp): fp for fp in glob(os.path.join(KS2_CSV_DIR, "*.csv"))
}
KS2_CANONICALIZED_CSV_DIR = os.path.join(INTERIM_DIR, "ks2_canonicalized_csv")
KS2_CANONICALIZED_CSV_FPS = {
    date: os.path.join(KS2_CANONICALIZED_CSV_DIR, f"ks2_{date}.csv")
    for date in KS2_CSV_FPS.keys()
}
KS2_MERGED_FP = os.path.join(INTERIM_DIR, "ks2_merged.csv")
KS2_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f"ks2_annotated.csv")

KS2_CSV_FP = os.path.join(INTERIM_DIR, f"ks2.csv")  # This is the final ks2 that pulls from ks4 and ks2 data

# NEET Files
NEET_DATASET_CSV_DIR = os.path.join(RAW_DATA_DIR, "ccis_original_csv")
NEET_DATASET_CSV_FPS = {
    _read_savedate(fp): fp for fp in glob(os.path.join(NEET_DATASET_CSV_DIR, "*.csv"))
}
NEET_CANONICALIZED_CSV_DIR = os.path.join(INTERIM_DIR, "ccis_canonicalized_csv")
NEET_CANONICALIZED_CSV_FPS = {
    date: os.path.join(NEET_CANONICALIZED_CSV_DIR, f"neet_{date}.csv")
    for date in NEET_DATASET_CSV_FPS.keys()
}
NEET_MERGED_FP = os.path.join(INTERIM_DIR, "neet_merged.csv")
NEET_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f"neet_annotated.csv")
NEET_PREMERGE_CSV_FP = os.path.join(INTERIM_DIR, f"neet_premerge.csv")

# Characteristics Files
CHARACTERISTICS_CSV_DIR = os.path.join(RAW_DATA_DIR, "characteristics_original_csv")
CHARACTERISTICS_CSV_FPS = {
    _read_savedate(fp): fp for fp in glob(os.path.join(CHARACTERISTICS_CSV_DIR, "*.csv"))
}
CHARACTERISTICS_CANONICALIZED_CSV_DIR = os.path.join(INTERIM_DIR, "characteristics_canonicalized_csv")
CHARACTERISTICS_CANONICALIZED_CSV_FPS = {
    date: os.path.join(CHARACTERISTICS_CANONICALIZED_CSV_DIR, f"characteristics_{date}.csv")
    for date in CHARACTERISTICS_CSV_FPS.keys()
}
CHARACTERISTICS_MERGED_FP = os.path.join(INTERIM_DIR, "characteristics_merged.csv")
CHARACTERISTICS_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f"characteristics_annotated.csv")
CHARACTERISTICS_PREMERGE_CSV_FP = os.path.join(INTERIM_DIR, f"characteristics_premerge.csv")


DATA_CODES_FP = os.path.join(RAW_DATA_DIR, "data_codes.csv")
