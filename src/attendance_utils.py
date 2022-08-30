"""
This utils file contains helpers for processing the attendance dataset.
"""

from dataclasses import asdict
import pandas as pd
import os

# from . import constants as c
# from . import school_info_utils as su
from .constants import (
    AttendanceDataColumns,
    AbsenceReasons,
    NonabsenceReasons,
    AuthorisedAbsenceReasons,
    UnauthorisedAbsenceReasons,
    ApprovedActivities,
)


def keep_column_criteria(col_name, gby_columns):
    if col_name in asdict(AbsenceReasons).values():
        return True
    if col_name in asdict(NonabsenceReasons).values():
        return True
    if col_name in [
        AttendanceDataColumns.late_present,
        AttendanceDataColumns.termly_sessions_possible,
    ]:
        return True
    if col_name in [
        AttendanceDataColumns.total_absences,
        AttendanceDataColumns.total_nonabsences,
    ]:
        return True
    if col_name in gby_columns:
        return True
    return False


def get_authorised_reason_columns(df):
    return [c for c in asdict(AuthorisedAbsenceReasons).values() if c in df.columns]


def get_unauthorised_reason_columns(df):
    return [c for c in asdict(UnauthorisedAbsenceReasons).values() if c in df.columns]


def get_approved_reason_columns(df):
    return [c for c in asdict(ApprovedActivities).values() if c in df.columns]


def get_absence_reason_columns(df):
    return [c for c in asdict(AbsenceReasons).values() if c in df.columns]


def get_nonabsence_reason_columns(df):
    return [c for c in asdict(NonabsenceReasons).values() if c in df.columns]
