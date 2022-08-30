"""
This file contains the the params for the data pipeline.
Before passing to DVC, it is converted into a yaml file
using scripts/generate_params_yaml.py. It supports native
python types as well as instances of classes.
"""
from ..constants import Targets, AttendanceTypes, FeatureSelectionMethods

PRE_AGE15 = True  # Should we drop students >= 15yrs old
TARGET = Targets.neet_ever
INCLUDE_TEST_TAKEN_CODE = True
ATTENDANCE_TYPE = AttendanceTypes.percent1

FEATURE_SELECTION_METHOD = FeatureSelectionMethods.remove_mostly_missing
REMOVE_MOSTLY_MISSING_MULTI_UPN_THRESHOLD = (
    0.6  # If we are doing remove_mostly_missing as our selection method
)
REMOVE_MOSTLY_MISSING_SINGLE_UPN_THRESHOLD = 0.7
FORWARD_FILL_FSME_COLUMN = True

REMOVE_MOSTLY_MISSING_PRE_COVID_MULTI_UPN_THRESHOLD = 0.8
REMOVE_MOSTLY_MISSING_POST_COVID_MULTI_UPN_THRESHOLD = 0.6

REMOVE_MOSTLY_MISSING_PRE_COVID_SINGLE_UPN_THRESHOLD = 0.8
REMOVE_MOSTLY_MISSING_POST_COVID_SINGLE_UPN_THRESHOLD = 0.6
