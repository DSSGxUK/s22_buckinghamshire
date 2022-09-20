"""
This file stores the command-line arguments to the different
stages of the pipeline. DVC will take these dictionaries 
and unpack them into the proper command line arguments
for the scripts run at each stage.
"""
from ..constants import DatasetTypes, OutputDatasetTypes
from .data_pipeline_params import *
from .model_pipeline_params import *
from .filepaths import *

# DVC does not appear to be able to handle list-substitution into deps
# See https://github.com/iterative/dvc/issues/8171 and https://github.com/iterative/dvc/issues/7151.
# My approach to this is to avoid doing list-substitution by using the following patterns:
# 1. Each stage should only output one file (even if I've added functionality to output more).
# If there are multiple outputs necessary, we can run this in a `foreach` in dvc.yaml.
# The `xl_to_csv` stage in `scripts/dvc.yaml` is a good example of this.
# 2. If multiple inputs are needed, try to group them into directories. Then
# we can just use the directory as the dependency. This way there is only
# one dependency. The `merge_data` in `scripts/dvc.yaml` is a good example of this.
# 3. The `cmd` setting should use dict-unpacking to pass arguments to the script.
# This allows to pass lists and booleans to the scripts. If we also need to template
# a separate value for `deps` or `outs` we can put the "args" in a sub-dictionary.
# See `merge_data` for an example of this.
# XL_TO_CSV_INPUTS = (
#     list(ATTENDANCE_DATA_XL_FPS.values())
#     + list(NEET_DATASET_XL_FPS.values())
#     + list(SCHOOL_CENSUS_XL_FPS.values())
#     + [SCHOOL_INFO_XL_FP]
# )
# XL_TO_CSV_OUTPUTS = (
#     list(ATTENDANCE_DATA_CSV_FPS.values())
#     + list(NEET_DATASET_CSV_FPS.values())
#     + list(SCHOOL_CENSUS_CSV_FPS.values())
#     + [SCHOOL_INFO_CSV_FP]
# )
# assert len(XL_TO_CSV_INPUTS) == len(XL_TO_CSV_OUTPUTS)

# XL_TO_CSV_ARGS = [
#     {"inputs": input_fp, "outputs": output_fp}
#     for input_fp, output_fp in zip(XL_TO_CSV_INPUTS, XL_TO_CSV_OUTPUTS)
# ]

CANONICALIZE_CSV_ARGS = (
    [
        {
            "inputs": input_fp,
            "outputs": output_fp,
            "dataset_type": DatasetTypes.attendance,
        }
        for input_fp, output_fp in zip(
            ATTENDANCE_DATA_CSV_FPS.values(), ATTENDANCE_CANONICALIZED_CSV_FPS.values()
        )
    ]
    + [
        {"inputs": input_fp, "outputs": output_fp, "dataset_type": DatasetTypes.census}
        for input_fp, output_fp in zip(
            SCHOOL_CENSUS_CSV_FPS.values(), SCHOOL_CENSUS_CANONICALIZED_CSV_FPS.values()
        )
    ]
    + [
        {"inputs": input_fp, "outputs": output_fp, "dataset_type": DatasetTypes.ccis}
        for input_fp, output_fp in zip(
            NEET_DATASET_CSV_FPS.values(), NEET_CANONICALIZED_CSV_FPS.values()
        )
    ]
    + [
        {"inputs": input_fp, "outputs": output_fp, "dataset_type": DatasetTypes.ks4}
        for input_fp, output_fp in zip(
            KS4_DATA_CSV_FPS.values(), KS4_CANONICALIZED_CSV_FPS.values()
        )
    ]
    + [
        {
            "inputs": SCHOOL_INFO_CSV_FP,
            "outputs": SCHOOL_INFO_CANONICALIZED_CSV_FP,
            "dataset_type": DatasetTypes.school_info,
        }
    ] + [
         {"inputs": input_fp, "outputs": output_fp, "dataset_type": DatasetTypes.ks2}
        for input_fp, output_fp in zip(
            KS2_CSV_FPS.values(), KS2_CANONICALIZED_CSV_FPS.values()
        )
    ] + [
         {"inputs": input_fp, "outputs": output_fp, "dataset_type": DatasetTypes.characteristics}
        for input_fp, output_fp in zip(
            CHARACTERISTICS_CSV_FPS.values(), CHARACTERISTICS_CANONICALIZED_CSV_FPS.values()
        )
    ]
)

MERGE_CENSUS_DATA_ARGS = {
    "inputs": list(SCHOOL_CENSUS_CANONICALIZED_CSV_FPS.values()),
    "output": SCHOOL_CENSUS_MERGED_FP,
    "data_dates": list(SCHOOL_CENSUS_CANONICALIZED_CSV_FPS.keys()),
    "dataset_type": DatasetTypes.census,
}
MERGE_ATTENDANCE_DATA_ARGS = {
    "inputs": list(ATTENDANCE_CANONICALIZED_CSV_FPS.values()),
    "output": ATTENDANCE_MERGED_FP,
    "data_dates": list(ATTENDANCE_CANONICALIZED_CSV_FPS.keys()),
    "dataset_type": DatasetTypes.attendance,
}
MERGE_CCIS_DATA_ARGS = {
    "inputs": list(NEET_CANONICALIZED_CSV_FPS.values()),
    "output": NEET_MERGED_FP,
    "data_dates": list(NEET_CANONICALIZED_CSV_FPS.keys()),
    "dataset_type": DatasetTypes.ccis,
}
MERGE_KS4_DATA_ARGS = {
    "inputs": list(KS4_CANONICALIZED_CSV_FPS.values()),
    "output": KS4_MERGED_FP,
    "data_dates": list(KS4_CANONICALIZED_CSV_FPS.keys()),
    "dataset_type": DatasetTypes.ks4,
}
MERGE_KS2_DATA_ARGS = {
    "inputs": list(KS2_CANONICALIZED_CSV_FPS.values()),
    "output": KS2_MERGED_FP,
    "data_dates": list(KS2_CANONICALIZED_CSV_FPS.keys()),
    "dataset_type": DatasetTypes.ks2,
}
MERGE_CHARACTERISTICS_DATA_ARGS = {
    "inputs": list(CHARACTERISTICS_CANONICALIZED_CSV_FPS.values()),
    "output": CHARACTERISTICS_MERGED_FP,
    "data_dates": list(CHARACTERISTICS_CANONICALIZED_CSV_FPS.keys()),
    "dataset_type": DatasetTypes.characteristics,
}

ANNOTATE_CENSUS_ARGS = {
    "input": SCHOOL_CENSUS_MERGED_FP,
    "school_info": SCHOOL_INFO_CANONICALIZED_CSV_FP,
    "output": SCHOOL_CENSUS_ANNOTATED_CSV_FP,
}
ANNOTATE_ATTENDANCE_ARGS = {
    "input": ATTENDANCE_MERGED_FP,
    "school_info": SCHOOL_INFO_CANONICALIZED_CSV_FP,
    "output": ATTENDANCE_ANNOTATED_CSV_FP,
}
ANNOTATE_CCIS_ARGS = {
    "input": NEET_MERGED_FP,
    "school_info": SCHOOL_INFO_CANONICALIZED_CSV_FP,
    "output": NEET_ANNOTATED_CSV_FP,
}
ANNOTATE_KS4_ARGS = {
    "input": KS4_MERGED_FP,
    "school_info": SCHOOL_INFO_CANONICALIZED_CSV_FP,
    "output": KS4_ANNOTATED_CSV_FP,
}
ANNOTATE_KS2_ARGS = {
    "input": KS2_MERGED_FP,
    "school_info": SCHOOL_INFO_CANONICALIZED_CSV_FP,
    "output": KS2_ANNOTATED_CSV_FP,
}
ANNOTATE_CHARACTERISTICS_ARGS = {
    "input": CHARACTERISTICS_MERGED_FP,
    "school_info": SCHOOL_INFO_CANONICALIZED_CSV_FP,
    "output": CHARACTERISTICS_ANNOTATED_CSV_FP,
}

CCIS_PREMERGE_ARGS = {"input": NEET_ANNOTATED_CSV_FP, "output": NEET_PREMERGE_CSV_FP}
CHARACTERISTICS_PREMERGE_ARGS = {
    "input": CHARACTERISTICS_ANNOTATED_CSV_FP,
    "output": CHARACTERISTICS_PREMERGE_CSV_FP
}
CENSUS_PREMERGE_ARGS = {
    "input": SCHOOL_CENSUS_ANNOTATED_CSV_FP,
    "output": SCHOOL_CENSUS_PREMERGE_CSV_FP,
}
ATTENDANCE_EXACT_ARGS = {
    "input": ATTENDANCE_ANNOTATED_CSV_FP,
    "output": ATTENDANCE_EXACT_CSV_FP,
    "attendance_type": AttendanceTypes.exact,
}
ATTENDANCE_PERCENT1_ARGS = {
    "input": ATTENDANCE_ANNOTATED_CSV_FP,
    "output": ATTENDANCE_PERCENT1_CSV_FP,
    "attendance_type": AttendanceTypes.percent1,
}
ATTENDANCE_PERCENT2_ARGS = {
    "input": ATTENDANCE_ANNOTATED_CSV_FP,
    "output": ATTENDANCE_PERCENT2_CSV_FP,
    "attendance_type": AttendanceTypes.percent2,
}
ATTENDANCE_NORMED_ARGS = {
    "input": ATTENDANCE_ANNOTATED_CSV_FP,
    "output": ATTENDANCE_NORMED_CSV_FP,
    "attendance_type": AttendanceTypes.term_normalized,
}
KS2_FILTER_ARGS = {
    "ks4_input": KS4_ANNOTATED_CSV_FP, 
    "ks2_input": KS2_ANNOTATED_CSV_FP,
    "output": KS2_CSV_FP}

MERGE_MULTI_MODELING_ARGS = {
    "ccis": NEET_PREMERGE_CSV_FP,
    "census": SCHOOL_CENSUS_PREMERGE_CSV_FP,
    "att": ATTENDANCE_PERCENT1_CSV_FP,
    "ks2": KS2_CSV_FP,
    "target": Targets.neet_ever,
    "output": MULTI_UPN_FP,
    "output_dataset_type": OutputDatasetTypes.modeling,
}

MERGE_MULTI_MODELING_BASIC_ARGS = {
    "ccis": NEET_PREMERGE_CSV_FP,
    "census": SCHOOL_CENSUS_PREMERGE_CSV_FP,
    "att": ATTENDANCE_PERCENT1_CSV_FP,
    "ks2": KS2_CSV_FP,
    "target": Targets.neet_ever,
    "output": MULTI_UPN_BASIC_FP,
    "output_dataset_type": OutputDatasetTypes.modeling,
    "only_att_census_merge": True
}

MERGE_MULTI_UNKNOWN_ARGS = {
    "ccis": NEET_PREMERGE_CSV_FP,
    "census": SCHOOL_CENSUS_PREMERGE_CSV_FP,
    "att": ATTENDANCE_PERCENT1_CSV_FP,
    "ks2": KS2_CSV_FP,
    "output": MULTI_UPN_UNKNOWNS_FP,
    "output_dataset_type": OutputDatasetTypes.unknowns,
}
MERGE_MULTI_PREDICTION_ARGS = {
    "ccis": NEET_PREMERGE_CSV_FP,
    "census": SCHOOL_CENSUS_PREMERGE_CSV_FP,
    "att": ATTENDANCE_PERCENT1_CSV_FP,
    "ks2": KS2_CSV_FP,
    "chars": CHARACTERISTICS_PREMERGE_CSV_FP,
    "output": MULTI_UPN_PREDICTION_FP,
    "output_dataset_type": OutputDatasetTypes.prediction,
}

MULTI_UPN_CATEGORICAL_ARGS = {
    "input": MULTI_UPN_FP,
    "output": MULTI_UPN_CATEGORICAL_FP,
    "include_test_taken_code": True,
    "output_dataset_type": OutputDatasetTypes.modeling,
}

MULTI_UPN_CATEGORICAL_BASIC_ARGS = {
    "input": MULTI_UPN_BASIC_FP,
    "output": MULTI_UPN_CATEGORICAL_BASIC_FP,
    "include_test_taken_code": False,
    "output_dataset_type": OutputDatasetTypes.modeling,
    "only_att_census_merge" : True
}

MULTI_UPN_CATEGORICAL_PREDICTIONS_ARGS = {
    "input": MULTI_UPN_PREDICTION_FP,
    "output": MULTI_UPN_CATEGORICAL_PREDICT_FP,
    "include_test_taken_code": True,
    "output_dataset_type": OutputDatasetTypes.prediction,
}

MULTI_UPN_CATEGORICAL_UNKNOWNS_ARGS = {
    "input": MULTI_UPN_UNKNOWNS_FP,
    "output": MULTI_UPN_CATEGORICAL_UNKNOWNS_FP,
    "include_test_taken_code": True,
    "output_dataset_type": OutputDatasetTypes.unknowns,
}

MULTIPLE_TO_SINGLE_ARGS = {
    "input": MULTI_UPN_CATEGORICAL_FP,
    "output": SINGLE_UPN_CATEGORICAL_FP,
}

MULTIPLE_TO_SINGLE_BASIC_ARGS = {
    "input": MULTI_UPN_CATEGORICAL_BASIC_FP,
    "output": SINGLE_UPN_CATEGORICAL_BASIC_FP,
}

#MULTIPLE_TO_SINGLE_PREDICTIONS_ARGS = {
#    "input": MULTI_UPN_CATEGORICAL_PREDICT_FP,
#    "output": SINGLE_UPN_CATEGORICAL_PREDICT_FP,
#}
MULTIPLE_TO_SINGLE_PREDICTIONS_ARGS = {
    "input": MULTI_UPN_CATEGORICAL_PREDICT_FP,
    "output_chars": SINGLE_UPN_CATEGORICAL_PREDICT_FP,
    "output_no_chars": SINGLE_UPN_CATEGORICAL_PREDICT_BASIC_FP,

}
MULTIPLE_TO_SINGLE_UNKNOWNS_ARGS = {
    "input": MULTI_UPN_CATEGORICAL_UNKNOWNS_FP,
    "output": SINGLE_UPN_CATEGORICAL_UNKNOWNS_FP,
}

FEATURE_SELECTION_SINGLE_UPN_PARAMS = {
    "input": SINGLE_UPN_CATEGORICAL_FP,
    "output": FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_FP,
    "single": True,
    "forward_fill_fsme_column": False,
    "feature_selection_method": FEATURE_SELECTION_METHOD,
    "remove_mostly_missing_threshold": REMOVE_MOSTLY_MISSING_SINGLE_UPN_THRESHOLD,
}
FEATURE_SELECTION_SINGLE_UPN_BASIC_PARAMS = {
    "input": SINGLE_UPN_CATEGORICAL_BASIC_FP,
    "output": FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_BASIC_FP,
    "single": True,
    "forward_fill_fsme_column": False,
    "feature_selection_method": FEATURE_SELECTION_METHOD,
    "remove_mostly_missing_threshold": REMOVE_MOSTLY_MISSING_SINGLE_UPN_THRESHOLD,
}
FEATURE_SELECTION_MULTI_UPN_ARGS = {
    "input": MULTI_UPN_CATEGORICAL_FP,
    "output": FEATURE_SELECTED_MULTI_UPN_CATEGORICAL_FP,
    "single": False,
    "forward_fill_fsme_column": True,
    "feature_selection_method": FEATURE_SELECTION_METHOD,
    "remove_mostly_missing_threshold": REMOVE_MOSTLY_MISSING_MULTI_UPN_THRESHOLD,
}
ADDITIONAL_DATA_ARGS = {
    "single": True,
    "census_input": SCHOOL_CENSUS_ANNOTATED_CSV_FP,
    "att_input": ATTENDANCE_ANNOTATED_CSV_FP,
    "output": ADDITIONAL_DATA_FP,
}

FEATURE_SELECTION_SINGLE_UPN_PREDICT_PARAMS = {
    "input": SINGLE_UPN_CATEGORICAL_PREDICT_FP,
    "output": FS_SINGLE_UPN_CATEGORICAL_PREDICT_FP,
    "single": True,
    "fill_fsme": False,
    "train_data": SINGLE_TRAIN_FP,
    "unidentified_csv": SINGLE_UNIDENTIFIED_PRED_ORIG_FP,
    "student_names": ADDITIONAL_DATA_FP,
}
FEATURE_SELECTION_SINGLE_UPN_PREDICT_BASIC_PARAMS = {
    "input": SINGLE_UPN_CATEGORICAL_PREDICT_BASIC_FP,
    "output": FS_SINGLE_UPN_CATEGORICAL_PREDICT_BASIC_FP,
    "single": True,
    "fill_fsme": False,
    "train_data": SINGLE_TRAIN_BASIC_FP,
    "unidentified_csv": SINGLE_UNIDENTIFIED_PRED_BASIC_FP,
    "student_names": ADDITIONAL_DATA_FP,
} 

FEATURE_SELECTION_MULTI_UPN_PREDICT_PARAMS = {
    "input": MULTI_UPN_CATEGORICAL_PREDICT_FP,
    "output": FS_MULTI_UPN_CATEGORICAL_PREDICT_FP,
    "single": False,
    "fill_fsme": True,
    "train_data": MULTI_TRAIN_FP,
    "unidentified_csv": MULTI_UNIDENTIFIED_PRED_FP,
    "student_names": ADDITIONAL_DATA_FP,
}

FEATURE_SELECTION_SINGLE_UPN_UNKNOWNS_PARAMS = {
    "input": SINGLE_UPN_CATEGORICAL_UNKNOWNS_FP,
    "output": FS_SINGLE_UPN_CATEGORICAL_UNKNOWNS_FP,
    "single": True,
    "fill_fsme": False,
    "train_data": SINGLE_TRAIN_FP,
    "unidentified_csv": SINGLE_UNIDENTIFIED_UNKS_FP,
    "student_names": ADDITIONAL_DATA_FP,
}

SPLIT_DATA_SINGLE_ARGS = {
    "input": FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_FP,
    "split": TEST_SPLIT,
    "train_output": SINGLE_TRAIN_FP,
    "test_output": SINGLE_TEST_FP,
    "single": True,
    "target": TARGET,
}

SPLIT_DATA_SINGLE_BASIC_ARGS = {
    "input": FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_BASIC_FP,
    "split": TEST_SPLIT,
    "train_output": SINGLE_TRAIN_BASIC_FP,
    "test_output": SINGLE_TEST_BASIC_FP,
    "single": True,
    "target": TARGET,
} 

SPLIT_DATA_MULTI_ARGS = {
    "input": FEATURE_SELECTED_MULTI_UPN_CATEGORICAL_FP,
    "split": TEST_SPLIT,
    "train_output": MULTI_TRAIN_FP,
    "test_output": MULTI_TEST_FP,
    "single": False,
    "target": TARGET,
}

FEATURE_SELECTION_MULTI_UPN_UNKNOWNS_PARAMS = {
    "input": MULTI_UPN_CATEGORICAL_UNKNOWNS_FP,
    "output": FS_MULTI_UPN_CATEGORICAL_UNKNOWNS_FP,
    "single": False,
    "fill_fsme": True,
    "train_data": MULTI_TRAIN_FP,
    "unidentified_csv": MULTI_UNIDENTIFIED_UNKS_FP,
    "student_names": ADDITIONAL_DATA_FP,
}


# FEATURE_SELECTION_SINGLE_UPN_PARAMS = {
#     'input' : SINGLE_UPN_CATEGORICAL_FP,
#     'output': FEATURE_SELECTED_SINGLE_UPN_CATEGORICAL_FP,
#     'single' : True,
#     'forward_fill_fsme_column' : False,
#     'feature_selection_method' : FEATURE_SELECTION_METHOD,
#     'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_SINGLE_UPN_THRESHOLD}

# SPLIT_COVID_YEARS_PARAMS = {
#     'input': MULTI_UPN_CATEGORICAL_FP ,
#     'output_pre_covid':PRE_COVID_MULTI_UPN_CATEGORICAL_FP,
#     'output_post_covid': POST_COVID_MULTI_UPN_CATEGORICAL_FP }

# FEATURE_SELECTION_MULTI_UPN_PRE_COVID_PARAMS = {
#     'input' : PRE_COVID_MULTI_UPN_CATEGORICAL_FP,
#     'output': FS_PRE_COVID_MULTI_UPN_CATEGORICAL_FP,
#     'single' : False,
#     'forward_fill_fsme_column' : True,
#     'feature_selection_method' : FEATURE_SELECTION_METHOD,
#     'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_PRE_COVID_MULTI_UPN_THRESHOLD}

# FEATURE_SELECTION_MULTI_UPN_POST_COVID_PARAMS = {
#     'input' : POST_COVID_MULTI_UPN_CATEGORICAL_FP,
#     'output': FS_POST_COVID_MULTI_UPN_CATEGORICAL_FP,
#     'single' : False,
#     'forward_fill_fsme_column' : True,
#     'feature_selection_method' : FEATURE_SELECTION_METHOD,
#     'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_POST_COVID_MULTI_UPN_THRESHOLD}

# MULTIPLE_TO_SINGLE_PRE_COVID_PARAMS = {
#     'input' : PRE_COVID_MULTI_UPN_CATEGORICAL_FP,
#     'output': PRE_COVID_SINGLE_UPN_CATEGORICAL_FP}

# MULTIPLE_TO_SINGLE_POST_COVID_PARAMS = {
#     'input' : POST_COVID_MULTI_UPN_CATEGORICAL_FP,
#     'output': POST_COVID_SINGLE_UPN_CATEGORICAL_FP}

# FEATURE_SELECTION_SINGLE_UPN_PRE_COVID_PARAMS = {
#     'input' : PRE_COVID_SINGLE_UPN_CATEGORICAL_FP,
#     'output': FS_PRE_COVID_SINGLE_UPN_CATEGORICAL_FP,
#     'single' : True,
#     'forward_fill_fsme_column' : False,
#     'feature_selection_method' : FEATURE_SELECTION_METHOD,
#     'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_PRE_COVID_SINGLE_UPN_THRESHOLD}

# FEATURE_SELECTION_SINGLE_UPN_POST_COVID_PARAMS = {
#     'input' : POST_COVID_SINGLE_UPN_CATEGORICAL_FP,
#     'output': FS_POST_COVID_SINGLE_UPN_CATEGORICAL_FP,
#     'single' : True,
#     'forward_fill_fsme_column' : False,
#     'feature_selection_method' : FEATURE_SELECTION_METHOD,
#     'remove_mostly_missing_threshold' : REMOVE_MOSTLY_MISSING_POST_COVID_SINGLE_UPN_THRESHOLD}
