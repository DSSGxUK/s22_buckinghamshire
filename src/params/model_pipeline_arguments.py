"""
This file stores the command-line arguments to the different
stages of the pipeline. DVC will take these dictionaries 
and unpack them into the proper command line arguments
for the scripts run at each stage.
"""

from .data_pipeline_params import *
from .model_pipeline_params import *
from .filepaths import *

RONI_PARAMS = {
    "train_data": SINGLE_TRAIN_FP,
    "test_data": SINGLE_TEST_FP,
    "output_test_metrics": RONI_TEST_RESULTS,
    "target": TARGET,
}

RETRAIN_SINGLE_PARAMS = {
    "input": SINGLE_TRAIN_FP,
    "single": True,
    "model_metrics": [LGBM1_METRICS_SINGLE, LGBM2_METRICS_SINGLE],
    "target": TARGET,
    "model_output_best": MODEL_BEST_THRESH_SINGLE,
    "model_output_mean": MODEL_MEAN_THRESH_SINGLE,
}

TEST_SINGLE_PARAMS = {
    "input": SINGLE_TEST_FP,
    "single": True,
    "output": TEST_RESULTS_SINGLE_CSV_FP,
    "model_output": FINAL_MODEL_SINGLE_FP,
    "model_pkls": [MODEL_BEST_THRESH_SINGLE, MODEL_MEAN_THRESH_SINGLE],
    "target": TARGET,
}

PREDICT_SINGLE_PARAMS = {
    "input": FS_SINGLE_UPN_CATEGORICAL_PREDICT_FP,
    "single": True,
    "output": PREDICTIONS_CSV_FP_SINGLE,
    "model_pkl": FINAL_MODEL_SINGLE_FP,
    "roni_threshold": RONI_TEST_RESULTS,
    "additional_data": ADDITIONAL_DATA_FP,
}

UNKNOWNS_SINGLE_PARAMS = {
    "input": FS_SINGLE_UPN_CATEGORICAL_UNKNOWNS_FP,
    "single": True,
    "output": UNKNOWN_PREDICTIONS_CSV_FP_SINGLE,
    "model_pkl": FINAL_MODEL_SINGLE_FP,
    "roni_threshold": RONI_TEST_RESULTS,
    "additional_data": ADDITIONAL_DATA_FP,
}

OPTIMIZATION_AND_CV_LGBM1_ARGS = {
    "single": True,
    "space": "lgbm1",
    "parallel": True,
    "num_thresholds": 100,
    "input": SINGLE_TRAIN_FP,
    "target": Targets.neet_ever,
    "load_checkpoint": LOAD_CHECKPOINTS,
    "checkpoint": True,
    "log_results": True,
    "num_folds": 4,
}
OPTIMIZATION_AND_CV_LGBM2_ARGS = {
    "single": True,
    "space": "lgbm2",
    "parallel": True,
    "num_thresholds": 100,
    "input": SINGLE_TRAIN_FP,
    "target": Targets.neet_ever,
    "load_checkpoint": LOAD_CHECKPOINTS,
    "checkpoint": True,
    "log_results": True,
    "num_folds": 4,
}
