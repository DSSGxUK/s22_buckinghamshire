from dataclasses import dataclass

from .ccis_constants import *
from .census_constants import *
from .attendance_constants import *
from .ks_constants import *


@dataclass
class _DatasetTypes:
    attendance: str = "attendance"
    census: str = "census"
    ccis: str = "ccis"
    ks4: str = "ks4"
    school_info: str = "school_info"
    # These two are new datasets we will incorporate for the council
    ks2: str = "ks2"
    characteristics: str = "characteristics"


DatasetTypes = _DatasetTypes()

RENAME_DICT = {
    DatasetTypes.attendance: ATTENDANCE_COLUMN_RENAME,
    DatasetTypes.census: SCHOOL_CENSUS_COLUMN_RENAME,
    DatasetTypes.ks4: KS4_COLUMN_RENAME,
    DatasetTypes.ccis: CCIS_COLUMN_RENAME,
    DatasetTypes.school_info: SCHOOL_INFO_RENAME,
    DatasetTypes.characteristics: CCIS_COLUMN_RENAME,
    DatasetTypes.ks2: KS4_COLUMN_RENAME,
}


@dataclass
class _OutputDatasetTypes:
    modeling: str = "modeling"
    unknowns: str = "unknowns"
    prediction: str = "prediction"


OutputDatasetTypes = _OutputDatasetTypes()

# These are the different feature selection methods
# we allow. This class does not actually implement any
# feature selection methods, it simply specifies the
# different options that we accept.
@dataclass
class _FeatureSelectionMethods:
    none: str = "none"
    remove_mostly_missing: str = "remove_mostly_missing"

FeatureSelectionMethods = _FeatureSelectionMethods()

# The different aggregations we allow.
# This class does not actually implement any
# aggregations, it simply specifies the different
# options that we accept. Most of these are already
# understood by pandas. The others have relevant helper
# code in src/aggregation_utils.py
@dataclass
class _Aggregations:
    mean: str = "mean"  # understood by pandas
    median: str = "median"  # understood by pandas
    max: str = "max"  # understood by pandas
    min: str = "min"  # understood by pandas
    last: str = "last"  # understood by pandas
    categorical_max: str = "categorical_max"
    last_with_unknown: str = "last_with_unknown"

Aggregations = _Aggregations()

# This dictinoary specifies how we aggregate the
# CCIS data when preparing it for merge into the
# multi-upn dataset (rows correspond to student-years).
NEET_PREMERGE_AGG = {
    CCISDataColumns.level_of_need_code: Aggregations.categorical_max,
    CCISDataColumns.characteristic_code: Aggregations.categorical_max,
    CCISDataColumns.ethnicity: Aggregations.last_with_unknown,
    CCISDataColumns.gender: Aggregations.last_with_unknown,
    CCISDataColumns.sen_support_flag: Aggregations.categorical_max,
    CCISDataColumns.send_flag: Aggregations.categorical_max,
    CCISDataColumns.neet_ever: Aggregations.max,
    CCISDataColumns.unknown_ever: Aggregations.max,
    CCISDataColumns.birth_year: Aggregations.last,
    CCISDataColumns.birth_month: Aggregations.last,
    CCISDataColumns.birth_month: Aggregations.last,
    CCISDataColumns.compulsory_school_always: Aggregations.min,
    CCISDataColumns.unknown_currently: Aggregations.max,
}

# This dictionary specifies how we aggregate the
# different features when converting from the
# multi-upn dataset (rows correspond to student-years)
# to the single-upn dataset (rows correspond to single students).
MULTI_UPN_CATEGORICAL_TO_SINGLE_AGGS = {
    CharacteristicsDataColumns.send_flag: Aggregations.categorical_max,
    CCISDataColumns.neet_ever: Aggregations.max,
    CCISDataColumns.neet_unknown_ever: Aggregations.max,
    CCISDataColumns.unknown_ever: Aggregations.max,
    CharacteristicsDataColumns.characteristic_code: Aggregations.categorical_max,
    CharacteristicsDataColumns.level_of_need_code: Aggregations.categorical_max,
    CensusDataColumns.fsme_on_census_day: Aggregations.max,
    AttendanceDataColumns.termly_sessions_possible: Aggregations.mean,
    AttendanceDataColumns.illness_authorised: Aggregations.mean,
    AttendanceDataColumns.medical_authorised: Aggregations.mean,
    AttendanceDataColumns.religious_authorised: Aggregations.mean,
    AttendanceDataColumns.study_leave_authorised: Aggregations.mean,
    AttendanceDataColumns.traveller_authorised: Aggregations.mean,
    AttendanceDataColumns.holiday_authorised: Aggregations.mean,
    AttendanceDataColumns.excluded_authorised: Aggregations.mean,
    AttendanceDataColumns.other_authorised: Aggregations.mean,
    AttendanceDataColumns.holiday_unauthorised: Aggregations.mean,
    AttendanceDataColumns.late_unauthorised: Aggregations.mean,
    AttendanceDataColumns.other_unauthorised: Aggregations.mean,
    AttendanceDataColumns.unknown_unauthorised: Aggregations.mean,
    AttendanceDataColumns.offsite_education_approved: Aggregations.mean,
    AttendanceDataColumns.interview_approved: Aggregations.mean,
    AttendanceDataColumns.late_present: Aggregations.mean,
    AttendanceDataColumns.sports_approved: Aggregations.mean,
    AttendanceDataColumns.educational_visit_approved: Aggregations.mean,
    AttendanceDataColumns.work_experience_approved: Aggregations.mean,
    AttendanceDataColumns.covid: Aggregations.mean,
    AttendanceDataColumns.exceptional_circumstances: Aggregations.mean,
    AttendanceDataColumns.total_absences: Aggregations.mean,
    AttendanceDataColumns.total_nonabsences: Aggregations.mean,
    KS2Columns.deprivation_indicator_idaci_score: Aggregations.last,
    KS2Columns.is_the_pupil_disadvantaged: Aggregations.max,
    KS2Columns.is_the_pupil_looked_after: Aggregations.max,
    KS2Columns.ks2_aps_va: Aggregations.mean,
    KS2Columns.ks2_aps_cubed: Aggregations.mean,
    KS2Columns.ks2_aps_squared: Aggregations.mean,
    KS2Columns.ks2_english_ps: Aggregations.mean,
    KS2Columns.ks2_english_ps_deviation: Aggregations.mean,
    KS2Columns.ks2_maths_ps: Aggregations.mean,
    KS2Columns.ks2_maths_ps_deviation: Aggregations.mean,
    KS2Columns.ks2_prior_band: Aggregations.mean,
    KS2Columns.ks2_reading_ps: Aggregations.mean,
    CensusDataColumns.enrol_status: Aggregations.categorical_max,
    SchoolInfoColumns.establishment_status: Aggregations.categorical_max,
    SchoolInfoColumns.establishment_type: Aggregations.categorical_max,
    ETHNICITY: Aggregations.categorical_max,
    GENDER: Aggregations.categorical_max,
    CensusDataColumns.language: Aggregations.categorical_max,
    CensusDataColumns.resourced_provision_indicator: Aggregations.categorical_max,
    CensusDataColumns.sen_unit_indicator: Aggregations.categorical_max,
    CensusDataColumns.sen_provision: Aggregations.categorical_max,
    CensusDataColumns.sen_need: Aggregations.categorical_max,
    KS2Columns.ks2_english: Aggregations.mean,
    KS2Columns.ks2_mathematics: Aggregations.mean,
    KS2Columns.ks2_english_ta: Aggregations.mean,
    KS2Columns.ks2_english_ta_level: Aggregations.mean,
    KS2Columns.ks2_english_level_finely_graded: Aggregations.mean,
    KS2Columns.ks2_mathematics_ta: Aggregations.mean,
    KS2Columns.ks2_mathematics_ta_level: Aggregations.mean,
    KS2Columns.ks2_mathematics_level_finely_graded: Aggregations.mean,
    KS2Columns.ks2_reading_ta_level: Aggregations.mean,
    KS2Columns.ks2_reading_level_finely_graded: Aggregations.mean,
    KS2Columns.ks2_mathematics_finely_graded: Aggregations.mean,
    KS2Columns.sen: Aggregations.categorical_max,
    KS2Columns.ks2_english + CODED_SUFFIX: Aggregations.categorical_max,
    KS2Columns.ks2_mathematics + CODED_SUFFIX: Aggregations.categorical_max,
    KS2Columns.ks2_english_ta + CODED_SUFFIX: Aggregations.categorical_max,
    KS2Columns.ks2_english_ta_level + CODED_SUFFIX: Aggregations.categorical_max,
    KS2Columns.ks2_english_level_finely_graded
    + CODED_SUFFIX: Aggregations.categorical_max,
    KS2Columns.ks2_mathematics_ta + CODED_SUFFIX: Aggregations.categorical_max,
    KS2Columns.ks2_mathematics_ta_level + CODED_SUFFIX: Aggregations.categorical_max,
    KS2Columns.ks2_mathematics_level_finely_graded
    + CODED_SUFFIX: Aggregations.categorical_max,
    KS2Columns.ks2_reading_ta_level + CODED_SUFFIX: Aggregations.categorical_max,
    KS2Columns.ks2_reading_level_finely_graded
    + CODED_SUFFIX: Aggregations.categorical_max,
    KS2Columns.ks2_mathematics_finely_graded
    + CODED_SUFFIX: Aggregations.categorical_max,
    CharacteristicsDataColumns.birth_month: Aggregations.categorical_max,
    CharacteristicsDataColumns.sen_support_flag: Aggregations.categorical_max,
    CensusDataColumns.has_census_data: Aggregations.max,
    KS2Columns.has_ks2_data: Aggregations.max,
    AttendanceDataColumns.has_attendance_data: Aggregations.max,
    CharacteristicsDataColumns.has_characteristics_data: Aggregations.max
}

YEAR_OF_COVID_SPLIT = 2018
