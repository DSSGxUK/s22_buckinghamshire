from subprocess import call

import pandas as pd

from src.constants import (
    CCISDataColumns
)

SINGLE_INPUT_DATA_FP = 'test_data/test_split_data_single_input.csv'
EXPECTED_SINGLE_TRAIN_OUTPUT_DATA_FP = 'test_data/test_split_data_single_train_output.csv'
EXPECTED_SINGLE_TEST_OUTPUT_DATA_FP = 'test_data/test_split_data_single_test_output.csv'
ACTUAL_SINGLE_TRAIN_OUTPUT_DATA_FP = 'tmp/train_single_upn.csv'
ACTUAL_SINGLE_TEST_OUTPUT_DATA_FP = 'tmp/test_single_upn.csv'

MULTI_INPUT_DATA_FP = "test_data/test_split_data_multi_input.csv"
EXPECTED_MULTI_TRAIN_OUTPUT_DATA_FP = "test_data/test_split_data_multi_train_output.csv"
EXPECTED_MULTI_TEST_OUTPUT_DATA_FP = "test_data/test_split_data_multi_test_output.csv"
ACTUAL_MULTI_TRAIN_OUTPUT_DATA_FP = "tmp/test_split_data_multi_train_output.csv"
ACTUAL_MULTI_TEST_OUTPUT_DATA_FP = "tmp/test_split_data_multi_test_output.csv"

def test_single_split_data_basic():
    call([
        "python", "../scripts/model/split_data.py", 
        "--input", SINGLE_INPUT_DATA_FP, 
        "--split", "0.2",
        "--test_output", ACTUAL_SINGLE_TEST_OUTPUT_DATA_FP,
        "--train_output", ACTUAL_SINGLE_TRAIN_OUTPUT_DATA_FP,
        "--seed", "1",
        "--no_file_logging",
        "--single",
        "--target", CCISDataColumns.neet_ever
    ])

    expected_train_output = pd.read_csv(EXPECTED_SINGLE_TRAIN_OUTPUT_DATA_FP, keep_default_na=False).convert_dtypes()
    expected_test_output = pd.read_csv(EXPECTED_SINGLE_TEST_OUTPUT_DATA_FP, keep_default_na=False).convert_dtypes()
    actual_train_output = pd.read_csv(ACTUAL_SINGLE_TRAIN_OUTPUT_DATA_FP, keep_default_na=False).convert_dtypes()
    actual_test_output = pd.read_csv(ACTUAL_SINGLE_TEST_OUTPUT_DATA_FP, keep_default_na=False).convert_dtypes()

    pd.testing.assert_frame_equal(expected_train_output, actual_train_output)
    pd.testing.assert_frame_equal(expected_test_output, actual_test_output)

def test_multi_split_data_basic():
    call([
        "python", "../scripts/model/split_data.py", 
        "--input", MULTI_INPUT_DATA_FP, 
        "--split", "0.2",
        "--test_output", ACTUAL_MULTI_TEST_OUTPUT_DATA_FP,
        "--train_output", ACTUAL_MULTI_TRAIN_OUTPUT_DATA_FP,
        "--seed", "1",
        "--no_file_logging",
        "--target", CCISDataColumns.neet_ever
    ])

    expected_train_output = pd.read_csv(EXPECTED_MULTI_TRAIN_OUTPUT_DATA_FP, keep_default_na=False).convert_dtypes()
    expected_test_output = pd.read_csv(EXPECTED_MULTI_TEST_OUTPUT_DATA_FP, keep_default_na=False).convert_dtypes()
    actual_train_output = pd.read_csv(ACTUAL_MULTI_TRAIN_OUTPUT_DATA_FP, keep_default_na=False).convert_dtypes()
    actual_test_output = pd.read_csv(ACTUAL_MULTI_TEST_OUTPUT_DATA_FP, keep_default_na=False).convert_dtypes()

    pd.testing.assert_frame_equal(expected_train_output, actual_train_output)
    pd.testing.assert_frame_equal(expected_test_output, actual_test_output)