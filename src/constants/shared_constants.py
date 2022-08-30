"""
The values below are constants shared by the different datasets
and scripts that reference these constants. Constants are values 
that are fixed throughout the code and not expected to change.
"""

# This variable specifies what values are treated as missing data
NA_VALS = ["", "#VALUE!", "nan"]

# Useful Shared Columns
UPN = "upn"
ACTIVITY = "activity_code"
YEAR = "year"
AGE = "age"
NC_YEAR_ACTUAL = "nc_year_actual"
ETHNICITY = "ethnicity"
GENDER = "gender"
DATA_DATE = "data_date"

# Unknown Codes
ETHNICITY_UNKNOWN = ["REFU", "NOBT"]
GENDER_UNKNOWN = ["U", "W"]
UNKNOWN_CODES = {ETHNICITY: ETHNICITY_UNKNOWN, GENDER: GENDER_UNKNOWN}

# The separator used when expanding categorical columns
# (with `pd.get_dummies`). It should be something that
# will not already appear in the column names. That
# way it can reliably be used to split apart the original
# column name and category in the future
CATEGORICAL_SEP = "__"
