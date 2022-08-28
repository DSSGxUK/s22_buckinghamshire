"""
This file contains the filepaths of all the raw, interim and processed datasets that are used in the data pipeline as well as filepaths of csv outputs from the model
"""

import os

DATA_DIR = '../data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = '../models'
FINAL_MODELS_DIR = os.path.join(MODELS_DIR, 'final')
INTERIM_MODELS_DIR = os.path.join(MODELS_DIR, 'interim')
METRICS_DIR = '../metrics'
PLOTS_DIR = '../plots'
LOGS_DIR = '../logs'
TMP_DIR = '../tmp'
PARAMS_DIR = '../src/params'

DVC_YAML = './dvc.yaml'
PARAMS_YAML = './params.yaml'

MULTI_UPN_FP = os.path.join(INTERIM_DIR, f'multi_upn.csv')
MULTI_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'multi_upn_categorical.csv')
SINGLE_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'single_upn_categorical.csv')
FEATURE_SELECTED_MULTI_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'feature_selected_multi_upn_categorical.csv')
FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'feature_selected_single_upn_categorical.csv')

# EDA datasets
NEET_KS_EDA_FP = os.path.join(PROCESSED_DATA_DIR, "neet_ks_eda.csv")

# RONI
RONI_RESULTS = os.path.join(METRICS_DIR, 'roni_scores.csv')

# Separate datasets for pre and post covid years
PRE_COVID_MULTI_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'pre_covid_multi_upn_categorical.csv')
POST_COVID_MULTI_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'post_covid_multi_upn_categorical.csv')
FS_PRE_COVID_MULTI_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'fs_pre_covid_multi_upn_categorical.csv')
FS_POST_COVID_MULTI_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'fs_post_covid_multi_upn_categorical.csv')
PRE_COVID_SINGLE_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'pre_covid_single_upn_categorical.csv')
POST_COVID_SINGLE_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'post_covid_single_upn_categorical.csv')
FS_PRE_COVID_SINGLE_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'fs_pre_covid_single_upn_categorical.csv')
FS_POST_COVID_SINGLE_UPN_CATEGORICAL_FP = os.path.join(PROCESSED_DATA_DIR, f'fs_post_covid_single_upn_categorical.csv')

# Train Test Datasets for SingleUPN
SINGLE_TRAIN_FP=os.path.join(PROCESSED_DATA_DIR, f'train_singleUPN.csv')
SINGLE_TEST_FP=os.path.join(PROCESSED_DATA_DIR, f'test_singleUPN.csv')

# Train Test Datasets for MultiUPN
MULTI_TRAIN_FP=os.path.join(PROCESSED_DATA_DIR, f'train_multiUPN.csv')
MULTI_TEST_FP=os.path.join(PROCESSED_DATA_DIR, f'test_multiUPN.csv')

# Model tranining outputs
MODEL_FP = os.path.join(FINAL_MODELS_DIR, 'model.pkl')
PIPELINE_MODEL_FP = os.path.join(INTERIM_MODELS_DIR, 'pipeline_model.pkl')
BASE_MODEL_FP = os.path.join(INTERIM_MODELS_DIR, 'base_model.pkl')
CHECKPOINT_FP = os.path.join(INTERIM_MODELS_DIR, 'checkpoint.pkl')
CV_METRICS_FP = os.path.join(METRICS_DIR, 'cv.yaml')
CV_RESULTS_CSV_FP = os.path.join(METRICS_DIR, 'cv.csv')

# Training and test data
TRAIN_FP = os.path.join(PROCESSED_DATA_DIR, 'train.csv')


# Model tranining outputs
MODEL_FP = os.path.join(FINAL_MODELS_DIR, 'model.pkl')
PIPELINE_MODEL_FP = os.path.join(INTERIM_MODELS_DIR, 'pipeline_model.pkl')
BASE_MODEL_FP = os.path.join(INTERIM_MODELS_DIR, 'base_model.pkl')
CHECKPOINT_FP = os.path.join(INTERIM_MODELS_DIR, 'checkpoint.pkl')
CV_METRICS_FP = os.path.join(METRICS_DIR, 'cv.yaml')
CV_RESULTS_CSV_FP = os.path.join(METRICS_DIR, 'cv.csv')
TEST_RESULTS_CSV_FP = os.path.join(METRICS_DIR, 'test_results.csv')
PREDICTIONS_CSV_FP = os.path.join(METRICS_DIR, 'predictions.csv')

# School data Paths
SCHOOL_INFO_XL_FP = os.path.join(RAW_DATA_DIR, 'Bucks_Secondary_School_2022.xlsx')
SCHOOL_INFO_CSV_FP = os.path.join(RAW_DATA_DIR, 'Bucks_Secondary_School_2022.csv')
SCHOOL_INFO_CANONICALIZED_CSV_FP = os.path.join(INTERIM_DIR, 'secondary_schools.csv')

# School Census Paths
SCHOOL_CENSUS_17_XL_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan17_Hard_Coded.xlsx'
)
SCHOOL_CENSUS_18_XL_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan18_Hard_Coded.xlsx'
)
SCHOOL_CENSUS_19_XL_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan19_Hard_Coded.xlsx'
)
SCHOOL_CENSUS_20_XL_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan20_Hard_Coded.xlsx'
)
SCHOOL_CENSUS_21_XL_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan21_Hard_Coded.xlsx'
)
SCHOOL_CENSUS_22_XL_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan22_Hard_Coded.xlsx'
)
SCHOOL_CENSUS_XL_FPS = {
    'jan17': SCHOOL_CENSUS_17_XL_FP,
    'jan18': SCHOOL_CENSUS_18_XL_FP,
    'jan19': SCHOOL_CENSUS_19_XL_FP,
    'jan20': SCHOOL_CENSUS_20_XL_FP,
    'jan21': SCHOOL_CENSUS_21_XL_FP,
    'jan22': SCHOOL_CENSUS_22_XL_FP,
}
SCHOOL_CENSUS_17_CSV_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan17_Hard_Coded.csv'
)
SCHOOL_CENSUS_18_CSV_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan18_Hard_Coded.csv'
)
SCHOOL_CENSUS_19_CSV_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan19_Hard_Coded.csv'
)
SCHOOL_CENSUS_20_CSV_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan20_Hard_Coded.csv'
)
SCHOOL_CENSUS_21_CSV_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan21_Hard_Coded.csv'
)
SCHOOL_CENSUS_22_CSV_FP = os.path.join(
    RAW_DATA_DIR, 
    'school_census_context_Jan22_Hard_Coded.csv'
)
SCHOOL_CENSUS_CSV_FPS = {
    'jan17': SCHOOL_CENSUS_17_CSV_FP,
    'jan18': SCHOOL_CENSUS_18_CSV_FP,
    'jan19': SCHOOL_CENSUS_19_CSV_FP,
    'jan20': SCHOOL_CENSUS_20_CSV_FP,
    'jan21': SCHOOL_CENSUS_21_CSV_FP,
    'jan22': SCHOOL_CENSUS_22_CSV_FP,
}

SCHOOL_CENSUS_CANONICALIZED_CSV_FPS = {
    date: os.path.join(INTERIM_DIR, f'census_{date}.csv') for date in SCHOOL_CENSUS_CSV_FPS.keys()
}
SCHOOL_CENSUS_MERGED_DATA_FP = os.path.join(INTERIM_DIR, 'census_merged.csv')
SCHOOL_CENSUS_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f'census_annotated.csv')
SCHOOL_CENSUS_PREMERGE_CSV_FP = os.path.join(INTERIM_DIR, f'census_premerge.csv')

# Attendance filepaths
ATTENDANCE_DATA_AUT21_XL_FP = os.path.join(
    RAW_DATA_DIR,
    'school_census_attendance_Aut21_Hard_Coded.xlsx'
)
ATTENDANCE_DATA_SPR21_XL_FP = os.path.join(
    RAW_DATA_DIR,
    'school_census_attendance_Spr21_Hard_Coded.xlsx'
)
ATTENDANCE_DATA_SUM21_XL_FP = os.path.join(
    RAW_DATA_DIR,
    'school_census_attendance_Sum21_Hard_Coded.xlsx'
)
ATTENDANCE_DATA_OCT14_MAY15_XL_FP = os.path.join(
    RAW_DATA_DIR,
    'Attendance_Oct14-May15_to_Warwick.xlsx'
)
ATTENDANCE_DATA_OCT15_MAY16_XL_FP = os.path.join(
    RAW_DATA_DIR,
    'Attendance_Oct15-May16_to_Warwick.xlsx'
)
ATTENDANCE_DATA_OCT16_MAY17_XL_FP = os.path.join(
    RAW_DATA_DIR,
    'Attendance_Oct16-May17_to_Warwick.xlsx'
)
ATTENDANCE_DATA_OCT17_MAY18_XL_FP = os.path.join(
    RAW_DATA_DIR,
    'Attendance_Oct17-May18_to_Warwick.xlsx'
)
ATTENDANCE_DATA_OCT18_MAY15_XL_FP = os.path.join(
    RAW_DATA_DIR,
    'Attendance_Oct18-Jan20_to_Warwick.xlsx'
)
ATTENDANCE_DATA_OCT20_MAY22_XL_FP = os.path.join(
    RAW_DATA_DIR,
    'Attendance_Oct20-May22_to_Warwick.xlsx'
)

ATTENDANCE_DATA_XL_FPS = {
    'aut21': ATTENDANCE_DATA_AUT21_XL_FP,
    'spr21': ATTENDANCE_DATA_SPR21_XL_FP,
    'sum21': ATTENDANCE_DATA_SUM21_XL_FP,
    'oct14': ATTENDANCE_DATA_OCT14_MAY15_XL_FP,
    'oct15': ATTENDANCE_DATA_OCT15_MAY16_XL_FP,
    'oct16': ATTENDANCE_DATA_OCT16_MAY17_XL_FP,
    'oct17': ATTENDANCE_DATA_OCT17_MAY18_XL_FP,
    'oct18': ATTENDANCE_DATA_OCT18_MAY15_XL_FP,
    'oct20': ATTENDANCE_DATA_OCT20_MAY22_XL_FP,
}
OLD_ATTENDANCE_XL_FPS = {
    k: v for k,v in ATTENDANCE_DATA_XL_FPS.items() if k in {
        'aut21', 'spr21', 'sum21'
    }
}
NEW_ATTENDANCE_DATA_XL_FPS = {
    k: v for k,v in ATTENDANCE_DATA_XL_FPS.items() if k not in OLD_ATTENDANCE_XL_FPS
}

ATTENDANCE_DATA_AUT21_CSV_FP = os.path.join(
    RAW_DATA_DIR,
    'school_census_attendance_Aut21_Hard_Coded.csv'
)
ATTENDANCE_DATA_SPR21_CSV_FP = os.path.join(
    RAW_DATA_DIR,
    'school_census_attendance_Spr21_Hard_Coded.csv'
)
ATTENDANCE_DATA_SUM21_CSV_FP = os.path.join(
    RAW_DATA_DIR,
    'school_census_attendance_Sum21_Hard_Coded.csv'
)
OLD_ATTENDANCE_DATA_CSV_FPS = {
    'aut21': ATTENDANCE_DATA_AUT21_CSV_FP,
    'spr21': ATTENDANCE_DATA_SPR21_CSV_FP,
    'sum21': ATTENDANCE_DATA_SUM21_CSV_FP,
}

NEW_ATTENDANCE_DATA_CSV_FPS = {
    'jan15':  os.path.join(RAW_DATA_DIR, 'attendance_original_jan15.csv'),
    'jan16':  os.path.join(RAW_DATA_DIR, 'attendance_original_jan16.csv'),
    'jan17':  os.path.join(RAW_DATA_DIR, 'attendance_original_jan17.csv'),
    'jan18':  os.path.join(RAW_DATA_DIR, 'attendance_original_jan18.csv'),
    'jan19':  os.path.join(RAW_DATA_DIR, 'attendance_original_jan19.csv'),
    'jan20':  os.path.join(RAW_DATA_DIR, 'attendance_original_jan20.csv'),
    'jan21':  os.path.join(RAW_DATA_DIR, 'attendance_original_jan21.csv'),
    'jan22':  os.path.join(RAW_DATA_DIR, 'attendance_original_jan22.csv'),
    'may15':  os.path.join(RAW_DATA_DIR, 'attendance_original_may15.csv'),
    'may16':  os.path.join(RAW_DATA_DIR, 'attendance_original_may16.csv'),
    'may17':  os.path.join(RAW_DATA_DIR, 'attendance_original_may17.csv'),
    'may18':  os.path.join(RAW_DATA_DIR, 'attendance_original_may18.csv'),
    'may19':  os.path.join(RAW_DATA_DIR, 'attendance_original_may19.csv'),
    'may21':  os.path.join(RAW_DATA_DIR, 'attendance_original_may21.csv'),
    'may22':  os.path.join(RAW_DATA_DIR, 'attendance_original_may22.csv'),
    'oct14':  os.path.join(RAW_DATA_DIR, 'attendance_original_oct14.csv'),
    'oct15':  os.path.join(RAW_DATA_DIR, 'attendance_original_oct15.csv'),
    'oct16':  os.path.join(RAW_DATA_DIR, 'attendance_original_oct16.csv'),
    'oct17':  os.path.join(RAW_DATA_DIR, 'attendance_original_oct17.csv'),
    'oct18':  os.path.join(RAW_DATA_DIR, 'attendance_original_oct18.csv'),
    'oct19':  os.path.join(RAW_DATA_DIR, 'attendance_original_oct19.csv'),
    'oct20':  os.path.join(RAW_DATA_DIR, 'attendance_original_oct20.csv'),
    'oct21':  os.path.join(RAW_DATA_DIR, 'attendance_original_oct21.csv'),
}

ATTENDANCE_CANONICALIZED_CSV_FPS = {
    date: os.path.join(INTERIM_DIR, f'attendance_{date}.csv') for date in NEW_ATTENDANCE_DATA_CSV_FPS.keys()
}

ATTENDANCE_MERGED_FP = os.path.join(INTERIM_DIR, 'attendance_merged.csv')
ATTENDANCE_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f'attendance_annotated.csv')
ATTENDANCE_PREMERGE_CSV_FP = os.path.join(INTERIM_DIR, f'attendance_premerge.csv')

# KS4 Files
KS4_DATA_CSV_FPS = {
    '2015': os.path.join(RAW_DATA_DIR, 'KS4_2015_Hard_Coded.csv'),
    '2016': os.path.join(RAW_DATA_DIR, 'KS4_2016_Hard_Coded.csv'),
    '2017': os.path.join(RAW_DATA_DIR, 'KS4_2017_Hard_Coded.csv'),
    '2018': os.path.join(RAW_DATA_DIR, 'KS4_2018_Hard_Coded.csv'),
    '2019': os.path.join(RAW_DATA_DIR, 'KS4_2019_Hard_Coded.csv'),
}
KS4_DATA_MERGED_FP = os.path.join(INTERIM_DIR, 'ks4_merged.csv')
KS4_CANONICALIZED_CSV_FPS = {
    date: os.path.join(INTERIM_DIR, f'ks4_{date}.csv') for date in KS4_DATA_CSV_FPS.keys()
}
KS4_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f'ks4_annotated.csv')
KS2_CSV_FP = os.path.join(INTERIM_DIR, f'ks2.csv')

# NEET Files
NEET_DATA_DIR = RAW_DATA_DIR  # Use of NEET_DATA_DIR is deprecated. Please use RAW_DATA_DIR instead
NEET_ORIGINAL_DATASET_FPS = {
    'feb18': os.path.join(RAW_DATA_DIR, 'FEB18_NEET_MI.xlsx'),
    'feb19': os.path.join(RAW_DATA_DIR, 'FEB19_NEET_MI.xlsx'),
    'feb20': os.path.join(RAW_DATA_DIR, 'FEB20_NEET_MI.xlsx'),
    'feb21': os.path.join(RAW_DATA_DIR, 'FEB21_NEET_MI.xlsx'),
    'feb22': os.path.join(RAW_DATA_DIR, 'FEB22_NEET_MI.xlsx'),
}
NEET_NEW_DATASET_XL_FPS = {
    'feb16': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb16_with_YPID_to_Warwick.xlsx'),
    'feb17': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb17_with_YPID_to_Warwick.xlsx'),
    'feb18': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb18_with_YPID_to_Warwick.xlsx'),
    'feb19': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb19_with_YPID_to_Warwick.xlsx'),
    'feb20': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb20_with-YPID_to_Warwick.xlsx'),
    'feb21': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb21_with_YPID_to_Warwick.xlsx'),
    'feb22': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb22_with_YPID_to_Warwick.xlsx'),
}
NEET_NEW_DATASET_CSV_FPS = {
    'feb16': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb16_with_YPID_to_Warwick.csv'),
    'feb17': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb17_with_YPID_to_Warwick.csv'),
    'feb18': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb18_with_YPID_to_Warwick.csv'),
    'feb19': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb19_with_YPID_to_Warwick.csv'),
    'feb20': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb20_with_YPID_to_Warwick.csv'),
    'feb21': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb21_with_YPID_to_Warwick.csv'),
    'feb22': os.path.join(RAW_DATA_DIR, 'NEET_MI_Feb22_with_YPID_to_Warwick.csv'),
}

NEET_CANONICALIZED_CSV_FPS = {
    date: os.path.join(INTERIM_DIR, f'neet_{date}.csv') for date in NEET_NEW_DATASET_CSV_FPS.keys()
}
NEET_MERGED_FP = os.path.join(INTERIM_DIR, 'neet_merged.csv')
NEET_ANNOTATED_CSV_FP = os.path.join(INTERIM_DIR, f'neet_annotated.csv')
NEET_PREMERGE_CSV_FP = os.path.join(INTERIM_DIR, f'neet_premerge.csv')

DATA_CODES_FP = os.path.join(RAW_DATA_DIR, 'data_codes.csv')
