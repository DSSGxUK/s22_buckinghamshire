from subprocess import call

import pandas as pd

INPUT_DATA_FP = 'test_data/test_annotate_neet_data_input.csv'
EXPECTED_OUTPUT_DATA_FP = 'test_data/test_annotate_neet_data_expected_output.csv'
ACTUAL_OUTPUT_DATA_FP = 'tmp/test_annotate_neet_data_actual_ouput.csv'
SCHOOL_INFO_CANONICALIZED_INPUT_FP = 'test_data/test_secondary_schools_input.csv'

def test_annotate_neet_data_basic():
    call([
        "python", "../scripts/data/annotate_neet_data.py", 
        "--input", INPUT_DATA_FP, 
        "--school_info", SCHOOL_INFO_CANONICALIZED_INPUT_FP,
        "--output", ACTUAL_OUTPUT_DATA_FP,
        "--no_file_logging"
    ])

    expected_output = pd.read_csv(EXPECTED_OUTPUT_DATA_FP, dtype=str, keep_default_na=False)
    actual_output = pd.read_csv(ACTUAL_OUTPUT_DATA_FP, dtype=str, keep_default_na=False)

    pd.testing.assert_frame_equal(expected_output, actual_output)