"""
This file stores the command-line arguments to the different
stages of the pipeline. DVC will take these dictionaries 
and unpack them into the proper command line arguments
for the scripts run at each stage.
"""
from .data_pipeline_params import *
from .model_pipeline_params import *
from .filepaths import *

OLD_ATT_TO_CSV_PARAMS = {
    k: {
        'input': OLD_ATTENDANCE_XL_FPS[k],
        'output': OLD_ATTENDANCE_DATA_CSV_FPS[k]
    } for k in OLD_ATTENDANCE_XL_FPS.keys()
}

NEW_ATT_TO_CSV_PARAMS = {
    'input': NEW_ATTENDANCE_DATA_XL_FPS,
    'output': NEW_ATTENDANCE_DATA_CSV_FPS}

NEET_TO_CSV_PARAMS = {
    'input': NEET_NEW_DATASET_XL_FPS,
    'output': NEET_NEW_DATASET_CSV_FPS}

CENSUS_TO_CSV_PARAMS = {
    'input': SCHOOL_CENSUS_XL_FPS,
    'output': SCHOOL_CENSUS_CSV_FPS}

SCHOOL_INFO_CSV_PARAMS =  {
    'input': SCHOOL_INFO_XL_FP,
    'output': SCHOOL_INFO_CSV_FP}

CANONICALIZE_NEET_PARAMS = {
    'input': NEET_NEW_DATASET_CSV_FPS,
    'output': NEET_CANONICALIZED_CSV_FPS}

CANONICALIZE_CENSUS_PARAMS = {
    'input': SCHOOL_CENSUS_CSV_FPS,
    'output': SCHOOL_CENSUS_CANONICALIZED_CSV_FPS}

CANONICALIZE_KS4_PARAMS = {
    'input': KS4_DATA_CSV_FPS,
    'output': KS4_CANONICALIZED_CSV_FPS}

CANONICALIZE_ATTENDANCE_PARAMS = {
    'input_new': NEW_ATTENDANCE_DATA_CSV_FPS,
    'input_old': OLD_ATTENDANCE_DATA_CSV_FPS,
    'output': ATTENDANCE_CANONICALIZED_CSV_FPS}

CANONICALIZE_SCHOOL_INFO_PARAMS = {
    'input': SCHOOL_INFO_CSV_FP,
    'output': SCHOOL_INFO_CANONICALIZED_CSV_FP}

MERGE_NEET_PARAMS = {
    'input':NEET_CANONICALIZED_CSV_FPS,
    'output': NEET_MERGED_FP}

MERGE_ATTENDANCE_PARAMS = {
    'input':ATTENDANCE_CANONICALIZED_CSV_FPS,
    'output': ATTENDANCE_MERGED_FP}

MERGE_KS4_PARAMS = {
    'input':KS4_CANONICALIZED_CSV_FPS,
    'output': KS4_DATA_MERGED_FP}
        
MERGE_CENSUS_PARAMS = {
    'input':SCHOOL_CENSUS_CANONICALIZED_CSV_FPS,
    'output': SCHOOL_CENSUS_MERGED_DATA_FP
}

ANNOTATE_NEET_PARAMS = {
    'input': NEET_MERGED_FP,
    'school_info': SCHOOL_INFO_CANONICALIZED_CSV_FP,
    'output' : NEET_ANNOTATED_CSV_FP,
    'no_logging': False}

ANNOTATE_ATTENDANCE_PARAMS = {
    'input_att': ATTENDANCE_MERGED_FP,
    'input_school_info': SCHOOL_INFO_CANONICALIZED_CSV_FP,
    'output': ATTENDANCE_ANNOTATED_CSV_FP}

ANNOTATE_KS4_PARAMS = {
    'input_ks4': KS4_DATA_MERGED_FP,
    'input_school_info': SCHOOL_INFO_CANONICALIZED_CSV_FP,
    'output': KS4_ANNOTATED_CSV_FP}

ANNOTATE_CENSUS_PARAMS = {
    'input_census': SCHOOL_CENSUS_MERGED_DATA_FP,
    'input_school_info': SCHOOL_INFO_CANONICALIZED_CSV_FP,
    'output': SCHOOL_CENSUS_ANNOTATED_CSV_FP}

ATTENDANCE_PREMERGE_PARAMS = {
    'input':ATTENDANCE_ANNOTATED_CSV_FP,
    'output':ATTENDANCE_PREMERGE_CSV_FP,
    'attendance_type': ATTENDANCE_TYPE,
}

KS2_FILTER_PARAMS = {
    'input':KS4_ANNOTATED_CSV_FP,
    'output':KS2_CSV_FP
}

CENSUS_PREMERGE_PARAMS = {
    'input':SCHOOL_CENSUS_ANNOTATED_CSV_FP,
    'output':SCHOOL_CENSUS_PREMERGE_CSV_FP
}
BUILD_CCIS_KS_EDA_DATASET_ARGS = {
    "ccis_input": NEET_ANNOTATED_CSV_FP,
    "ks_input": KS4_ANNOTATED_CSV_FP,
    "output": NEET_KS_EDA_FP
}
NEET_PREMERGE_STAGE_PARAMS = {
    'input': NEET_ANNOTATED_CSV_FP,
    'output': NEET_PREMERGE_CSV_FP,
    'pre_age15': PRE_AGE15
}

MERGE_MULTI_UPN_PARAMS = {'output': MULTI_UPN_FP,
                        'att': ATTENDANCE_PREMERGE_CSV_FP,
                        'ks': KS2_CSV_FP,
                        'census': SCHOOL_CENSUS_PREMERGE_CSV_FP,
                        'ccis': NEET_PREMERGE_CSV_FP,
                        'target': TARGET,
                        'pre_age15':PRE_AGE15
}

MULTI_UPN_CATEGORICAL_PARAMS = {
    'input': MULTI_UPN_FP,
    'output': MULTI_UPN_CATEGORICAL_FP,
    'include_test_taken_code': True
}

FEATURE_SELECTION_MULTI_UPN_PARAMS = {
    'input' : MULTI_UPN_CATEGORICAL_FP,
    'output': FEATURE_SELECTED_MULTI_UPN_CATEGORICAL_FP,
    'single' : False,
    'forward_fill_fsme_column' : True, 
    'feature_selection_method' : FEATURE_SELECTION_METHOD,
    'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_MULTI_UPN_THRESHOLD}

MULTIPLE_TO_SINGLE_PARAMS = {
    'input' : MULTI_UPN_CATEGORICAL_FP,
    'output': SINGLE_UPN_CATEGORICAL_FP}

FEATURE_SELECTION_SINGLE_UPN_PARAMS = {
    'input' : SINGLE_UPN_CATEGORICAL_FP,
    'output': FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_FP,
    'single' : True,
    'forward_fill_fsme_column' : False, 
    'feature_selection_method' : FEATURE_SELECTION_METHOD,
    'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_SINGLE_UPN_THRESHOLD}

SPLIT_COVID_YEARS_PARAMS = {
    'input': MULTI_UPN_CATEGORICAL_FP , 
    'output_pre_covid':PRE_COVID_MULTI_UPN_CATEGORICAL_FP,
    'output_post_covid': POST_COVID_MULTI_UPN_CATEGORICAL_FP }

FEATURE_SELECTION_MULTI_UPN_PRE_COVID_PARAMS = {
    'input' : PRE_COVID_MULTI_UPN_CATEGORICAL_FP,
    'output': FS_PRE_COVID_MULTI_UPN_CATEGORICAL_FP,
    'single' : False,
    'forward_fill_fsme_column' : True, 
    'feature_selection_method' : FEATURE_SELECTION_METHOD,
    'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_PRE_COVID_MULTI_UPN_THRESHOLD}

FEATURE_SELECTION_MULTI_UPN_POST_COVID_PARAMS = {
    'input' : POST_COVID_MULTI_UPN_CATEGORICAL_FP,
    'output': FS_POST_COVID_MULTI_UPN_CATEGORICAL_FP,
    'single' : False,
    'forward_fill_fsme_column' : True, 
    'feature_selection_method' : FEATURE_SELECTION_METHOD,
    'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_POST_COVID_MULTI_UPN_THRESHOLD}

MULTIPLE_TO_SINGLE_PRE_COVID_PARAMS = {
    'input' : PRE_COVID_MULTI_UPN_CATEGORICAL_FP,
    'output': PRE_COVID_SINGLE_UPN_CATEGORICAL_FP}

MULTIPLE_TO_SINGLE_POST_COVID_PARAMS = {
    'input' : POST_COVID_MULTI_UPN_CATEGORICAL_FP,
    'output': POST_COVID_SINGLE_UPN_CATEGORICAL_FP}

FEATURE_SELECTION_SINGLE_UPN_PRE_COVID_PARAMS = {
    'input' : PRE_COVID_SINGLE_UPN_CATEGORICAL_FP,
    'output': FS_PRE_COVID_SINGLE_UPN_CATEGORICAL_FP,
    'single' : True,
    'forward_fill_fsme_column' : False, 
    'feature_selection_method' : FEATURE_SELECTION_METHOD,
    'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_PRE_COVID_SINGLE_UPN_THRESHOLD}

FEATURE_SELECTION_SINGLE_UPN_POST_COVID_PARAMS = {
    'input' : POST_COVID_SINGLE_UPN_CATEGORICAL_FP,
    'output': FS_POST_COVID_SINGLE_UPN_CATEGORICAL_FP,
    'single' : True,
    'forward_fill_fsme_column' : False, 
    'feature_selection_method' : FEATURE_SELECTION_METHOD,
    'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_POST_COVID_SINGLE_UPN_THRESHOLD}

SPLIT_DATA_SINGLE_PARAMS = {
    'input' : FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_FP,
    'split' : TEST_SPLIT,
    'train_output': SINGLE_TRAIN_FP,
    'test_output' : SINGLE_TEST_FP,
    'single' : True,
    'target' : TARGET    
}

SPLIT_DATA_MULTI_PARAMS = {
    'input' : FEATURE_SELECTED_MULTI_UPN_CATEGORICAL_FP,
    'split' : TEST_SPLIT,
    'train_output': MULTI_TRAIN_FP,
    'test_output' : MULTI_TEST_FP,
    'single' : False,
    'target' : TARGET    
}

