"""
This file contains the filepaths of all the raw, interim and processed datasets that are used in the data pipeline as well as filepaths of csv outputs from the model
"""

import os

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

DVC_YAML = "./dvc.yaml"
PARAMS_YAML = "./params.yaml"

MULTI_UPN_FP = os.path.join(INTERIM_DIR, f"multi_upn.csv")
MULTI_UPN_UNKNOWNS_FP = os.path.join(INTERIM_DIR, f"multi_upn_unknowns.csv")
MULTI_UPN_PREDICTION_FP = os.path.join(INTERIM_DIR, f"multi_upn_prediction.csv")
MULTI_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"multi_upn_categorical.csv"
)
SINGLE_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"single_upn_categorical.csv"
)
FEATURE_SELECTED_MULTI_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"feature_selected_multi_upn_categorical.csv"
)
FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_FP = os.path.join(
    PROCESSED_DATA_DIR, f"feature_selected_single_upn_categorical.csv"
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
CHECKPOINT_FP = os.path.join(INTERIM_MODELS_DIR, "checkpoint.pkl")
CV_METRICS_FP = os.path.join(METRICS_DIR, "cv.yaml")
CV_RESULTS_CSV_FP = os.path.join(METRICS_DIR, "cv.csv")
TEST_RESULTS_CSV_FP = os.path.join(METRICS_DIR, "test_results.csv")
RONI_TEST_RESULTS = os.path.join(METRICS_DIR, "roni_test_results.csv")

LGBM1_METRICS_MULTI = os.path.join(METRICS_DIR, "lgbm1.csv")
LGBM1_METRICS_SINGLE = os.path.join(METRICS_DIR, "lgbm_single.csv")

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
FINAL_MODEL_SINGLE_FP = os.path.join(FINAL_MODELS_DIR, "model_single.pkl")
TEST_RESULTS_SINGLE_CSV_FP = os.path.join(METRICS_DIR, "single_test_results.csv")

# Prediction & unknown datasets

SINGLE_UPN_CATEGORICAL_PREDICT_FP = os.path.join(
    PROCESSED_DATA_DIR, "predict_singleUPN.csv"
)
SINGLE_UPN_CATEGORICAL_UNKNOWNS_FP = os.path.join(
    PROCESSED_DATA_DIR, "unknowns_singleUPN.csv"
)
FS_SINGLE_UPN_CATEGORICAL_PREDICT_FP = os.path.join(
    PROCESSED_DATA_DIR, "fs_predict_singleUPN.csv"
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

# Unidentified students datasets
SINGLE_UNIDENTIFIED_PRED_FP = os.path.join(
    RESULTS_DIR, "unidentified_students_single.csv"
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
SCHOOL_INFO_XL_FP = os.path.join(RAW_DATA_DIR, "Bucks_Secondary_School_2022.xlsx")
SCHOOL_INFO_CSV_FP = os.path.join(RAW_DATA_DIR, "Bucks_Secondary_School_2022.csv")
SCHOOL_INFO_CANONICALIZED_CSV_FP = os.path.join(INTERIM_DIR, "secondary_schools.csv")

# School Census Paths
SCHOOL_CENSUS_XL_DIR = os.path.join(RAW_DATA_DIR, "census_original_xl")
SCHOOL_CENSUS_XL_FPS = {
    "jan17": os.path.join(
        SCHOOL_CENSUS_XL_DIR, "school_census_context_Jan17_Hard_Coded.xlsx"
    ),
    "jan18": os.path.join(
        SCHOOL_CENSUS_XL_DIR, "school_census_context_Jan18_Hard_Coded.xlsx"
    ),
    "jan19": os.path.join(
        SCHOOL_CENSUS_XL_DIR, "school_census_context_Jan19_Hard_Coded.xlsx"
    ),
    "jan20": os.path.join(
        SCHOOL_CENSUS_XL_DIR, "school_census_context_Jan20_Hard_Coded.xlsx"
    ),
    "jan21": os.path.join(
        SCHOOL_CENSUS_XL_DIR, "school_census_context_Jan21_Hard_Coded.xlsx"
    ),
    "jan22": os.path.join(
        SCHOOL_CENSUS_XL_DIR, "school_census_context_Jan22_Hard_Coded.xlsx"
    ),
}
SCHOOL_CENSUS_CSV_DIR = os.path.join(RAW_DATA_DIR, "census_original_csv")
SCHOOL_CENSUS_CSV_FPS = {
    "jan17": os.path.join(
        SCHOOL_CENSUS_CSV_DIR, "school_census_context_Jan17_Hard_Coded.csv"
    ),
    "jan18": os.path.join(
        SCHOOL_CENSUS_CSV_DIR, "school_census_context_Jan18_Hard_Coded.csv"
    ),
    "jan19": os.path.join(
        SCHOOL_CENSUS_CSV_DIR, "school_census_context_Jan19_Hard_Coded.csv"
    ),
    "jan20": os.path.join(
        SCHOOL_CENSUS_CSV_DIR, "school_census_context_Jan20_Hard_Coded.csv"
    ),
    "jan21": os.path.join(
        SCHOOL_CENSUS_CSV_DIR, "school_census_context_Jan21_Hard_Coded.csv"
    ),
    "jan22": os.path.join(
        SCHOOL_CENSUS_CSV_DIR, "school_census_context_Jan22_Hard_Coded.csv"
    ),
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
ATTENDANCE_XL_DIR = os.path.join(RAW_DATA_DIR, "attendance_original_xl")
ATTENDANCE_DATA_XL_FPS = {
    "jan15": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_jan15.xlsx"),
    "jan16": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_jan16.xlsx"),
    "jan17": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_jan17.xlsx"),
    "jan18": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_jan18.xlsx"),
    "jan19": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_jan19.xlsx"),
    "jan20": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_jan20.xlsx"),
    "jan21": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_jan21.xlsx"),
    "jan22": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_jan22.xlsx"),
    "may15": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_may15.xlsx"),
    "may16": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_may16.xlsx"),
    "may17": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_may17.xlsx"),
    "may18": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_may18.xlsx"),
    "may19": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_may19.xlsx"),
    "may21": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_may21.xlsx"),
    "may22": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_may22.xlsx"),
    "oct14": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_oct14.xlsx"),
    "oct15": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_oct15.xlsx"),
    "oct16": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_oct16.xlsx"),
    "oct17": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_oct17.xlsx"),
    "oct18": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_oct18.xlsx"),
    "oct19": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_oct19.xlsx"),
    "oct20": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_oct20.xlsx"),
    "oct21": os.path.join(ATTENDANCE_XL_DIR, "attendance_original_oct21.xlsx"),
}

ATTENDANCE_CSV_DIR = os.path.join(RAW_DATA_DIR, "attendance_original_csv")
ATTENDANCE_DATA_CSV_FPS = {
    "jan15": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_jan15.csv"),
    "jan16": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_jan16.csv"),
    "jan17": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_jan17.csv"),
    "jan18": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_jan18.csv"),
    "jan19": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_jan19.csv"),
    "jan20": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_jan20.csv"),
    "jan21": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_jan21.csv"),
    "jan22": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_jan22.csv"),
    "may15": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_may15.csv"),
    "may16": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_may16.csv"),
    "may17": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_may17.csv"),
    "may18": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_may18.csv"),
    "may19": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_may19.csv"),
    "may21": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_may21.csv"),
    "may22": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_may22.csv"),
    "oct14": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_oct14.csv"),
    "oct15": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_oct15.csv"),
    "oct16": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_oct16.csv"),
    "oct17": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_oct17.csv"),
    "oct18": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_oct18.csv"),
    "oct19": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_oct19.csv"),
    "oct20": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_oct20.csv"),
    "oct21": os.path.join(ATTENDANCE_CSV_DIR, "attendance_original_oct21.csv"),
}

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
    "sep15": os.path.join(KS4_CSV_DIR, "KS4_2015_Hard_Coded.csv"),
    "sep16": os.path.join(KS4_CSV_DIR, "KS4_2016_Hard_Coded.csv"),
    "sep17": os.path.join(KS4_CSV_DIR, "KS4_2017_Hard_Coded.csv"),
    "sep18": os.path.join(KS4_CSV_DIR, "KS4_2018_Hard_Coded.csv"),
    "sep19": os.path.join(KS4_CSV_DIR, "KS4_2019_Hard_Coded.csv"),
}
KS4_CANONICALIZED_CSV_DIR = os.path.join(INTERIM_DIR, "ks4_canonicalized_csv")
KS4_CANONICALIZED_CSV_FPS = {
    date: os.path.join(KS4_CANONICALIZED_CSV_DIR, f"ks4_{date}.csv")
    for date in KS4_DATA_CSV_FPS.keys()
}
KS4_MERGED_FP = os.path.join(INTERIM_DIR, "ks4_merged.csv")
KS4_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f"ks4_annotated.csv")
KS2_CSV_FP = os.path.join(INTERIM_DIR, f"ks2.csv")

# NEET Files
NEET_DATASET_XL_DIR = os.path.join(RAW_DATA_DIR, "ccis_original_xl")
NEET_DATASET_XL_FPS = {
    "mar16": os.path.join(
        NEET_DATASET_XL_DIR, "NEET_MI_Feb16_with_YPID_to_Warwick.xlsx"
    ),
    "mar17": os.path.join(
        NEET_DATASET_XL_DIR, "NEET_MI_Feb17_with_YPID_to_Warwick.xlsx"
    ),
    "mar18": os.path.join(
        NEET_DATASET_XL_DIR, "NEET_MI_Feb18_with_YPID_to_Warwick.xlsx"
    ),
    "mar19": os.path.join(
        NEET_DATASET_XL_DIR, "NEET_MI_Feb19_with_YPID_to_Warwick.xlsx"
    ),
    "mar20": os.path.join(
        NEET_DATASET_XL_DIR, "NEET_MI_Feb20_with-YPID_to_Warwick.xlsx"
    ),
    "mar21": os.path.join(
        NEET_DATASET_XL_DIR, "NEET_MI_Feb21_with_YPID_to_Warwick.xlsx"
    ),
    "mar22": os.path.join(
        NEET_DATASET_XL_DIR, "NEET_MI_Feb22_with_YPID_to_Warwick.xlsx"
    ),
}
NEET_DATASET_CSV_DIR = os.path.join(RAW_DATA_DIR, "ccis_original_csv")
NEET_DATASET_CSV_FPS = {
    "mar16": os.path.join(
        NEET_DATASET_CSV_DIR, "NEET_MI_Feb16_with_YPID_to_Warwick.csv"
    ),
    "mar17": os.path.join(
        NEET_DATASET_CSV_DIR, "NEET_MI_Feb17_with_YPID_to_Warwick.csv"
    ),
    "mar18": os.path.join(
        NEET_DATASET_CSV_DIR, "NEET_MI_Feb18_with_YPID_to_Warwick.csv"
    ),
    "mar19": os.path.join(
        NEET_DATASET_CSV_DIR, "NEET_MI_Feb19_with_YPID_to_Warwick.csv"
    ),
    "mar20": os.path.join(
        NEET_DATASET_CSV_DIR, "NEET_MI_Feb20_with_YPID_to_Warwick.csv"
    ),
    "mar21": os.path.join(
        NEET_DATASET_CSV_DIR, "NEET_MI_Feb21_with_YPID_to_Warwick.csv"
    ),
    "mar22": os.path.join(
        NEET_DATASET_CSV_DIR, "NEET_MI_Feb22_with_YPID_to_Warwick.csv"
    ),
}

NEET_CANONICALIZED_CSV_DIR = os.path.join(RAW_DATA_DIR, "ccis_canonicalized_csv")
NEET_CANONICALIZED_CSV_FPS = {
    date: os.path.join(NEET_CANONICALIZED_CSV_DIR, f"neet_{date}.csv")
    for date in NEET_DATASET_CSV_FPS.keys()
}
NEET_MERGED_FP = os.path.join(INTERIM_DIR, "neet_merged.csv")
NEET_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f"neet_annotated.csv")
NEET_PREMERGE_CSV_FP = os.path.join(INTERIM_DIR, f"neet_premerge.csv")

DATA_CODES_FP = os.path.join(RAW_DATA_DIR, "data_codes.csv")
