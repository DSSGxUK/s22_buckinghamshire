"""
This file contains the constants related to the Attendancqe datasets.
Constants are values that are fixed throughout the code and
not expected to change.

These constants are used by the other code in our project
when working with the data. We specify them below so that
if they need to change, we can simply change them below without
changing the rest of the code. Most of the constants are the column
names in the datasets.

The collections of different column names are specified
using dataclasses. Dataclasses are a shorthand for classes
of the form:

>>> class A:
...   def __init__(self):
...     self.a = 'hello'
...     self.b = 'goodbye'

written instead as

>>> @dataclass
... class A:
...   a: str = 'hello'
...   b: str = 'goodbye'

The advantage of using a dataclass is it hides the boilerplate
of ordinary classes. They also come with useful helpers (e.g.
asdict, which will create a dictionary from the fields of the
dataclass). For example

>>> asdict(A())
{'a': 'hello', 'b': 'goodbye'}
>>> list(asdict(A()).values())
['hello', 'goodbye']

Since the column names are constants, we only need to work with a single
instance of the dataclass. For that reason we write the class name with
an underscore in front, and assign an actual instance of the dataclass
the same name without the underscore. This pattern signifies that any code
should only use the variable name without the underscore in front. In our
example above:

>>> @dataclass
... class _A:
...   a: str = 'hello'
...   b: str = 'goodbye'

>>> A = _A()

Future code would then import and use `A`, not `_A`:

>>> from ... import A
>>> print(A.a)
'hello'

"""
from dataclasses import dataclass

from .shared_constants import *
from .school_info_constants import *

# Attendance
@dataclass
class _AuthorisedAbsenceReasons:
    """
    These are all the reasons for authorised absences.
    These are counted when computing the `total_absences`
    of a student.
    """

    excluded_authorised: str = "excluded_authorised"
    holiday_authorised: str = "holiday_authorised"
    illness_authorised: str = "illness_authorised"
    medical_authorised: str = "medical_authorised"
    other_authorised: str = "other_authorised"
    religious_authorised: str = "religious_authorised"
    study_leave_authorised: str = "study_leave_authorised"
    traveller_authorised: str = "traveller_authorised"


AuthorisedAbsenceReasons = _AuthorisedAbsenceReasons()


@dataclass
class _UnauthorisedAbsenceReasons:
    """
    These are all the reasons for unauthorised absences.
    These are counted when computing the `total_absences`
    of a student.
    """

    holiday_unauthorised: str = "holiday_unauthorised"
    late_unauthorised: str = "late_unauthorised"
    other_unauthorised: str = "other_unauthorised"

    unknown_unauthorised: str = "unknown_unauthorised"


UnauthorisedAbsenceReasons = _UnauthorisedAbsenceReasons()


@dataclass
class _ApprovedActivities:
    """
    These are all the reasons for approved educational activities.
    These are counted when computing the `total_absences`
    of a student.
    """

    educational_visit_approved: str = "educational_visit_approved"
    interview_approved: str = "interview_approved"
    offsite_education_approved: str = "offsite_education_approved"
    sports_approved: str = "sports_approved"
    work_experience_approved: str = "work_experience_approved"


ApprovedActivities = _ApprovedActivities()


@dataclass
class _AbsenceReasons(
    _AuthorisedAbsenceReasons, _UnauthorisedAbsenceReasons, _ApprovedActivities
):
    pass


AbsenceReasons = _AbsenceReasons()


@dataclass
class _NonabsenceReasons:
    """
    These are reasons for which the student does not attend
    school, but do not count as absences. The terms the student
    misses for the reasons below are deducted from `termly_sessions_possible`.
    """

    covid: str = "covid"
    exceptional_circumstances: str = "exceptional_circumstances"


NonabsenceReasons = _NonabsenceReasons()


@dataclass
class _AttendanceOriginalDataColumns(_AbsenceReasons, _NonabsenceReasons):
    code_f: str = "code_f"  # We couldn't figure out what this was

    upn: str = UPN
    date_of_birth: str = "date_of_birth"
    dual_registered: str = "dual_registered"
    enrol_status: str = "enrol_status"
    entry_date: str = "entry_date"
    establishment_number: str = SchoolInfoColumns.establishment_number
    nc_year_actual: str = NC_YEAR_ACTUAL
    forename: str = "forename"
    former_surname: str = "former_surname"
    middlenames: str = "middlenames"
    preferred_surname: str = "preferred_surname"
    surname: str = "surname"
    gender: str = GENDER

    # We ended up just summing up the reasons above rather than
    # using these two columns below because the two below seemed to be
    # innacurate. They seemed to greatly underestimate the actual number
    # of absences.
    termly_sessions_authorised: str = "termly_sessions_authorised"
    termly_sessions_unauthorised: str = "termly_sessions_unauthorised"

    # These are only available in later years.
    late_present: str = "late_present"
    present_am: str = "present_am"
    present_pm: str = "present_pm"

    termly_sessions_possible: str = "termly_sessions_possible"


AttendanceOriginalDataColumns = _AttendanceOriginalDataColumns()


@dataclass
class _AttendanceAddedDataColumns:
    # Added columns
    year: str = YEAR
    authorised_absences: str = "authorised_absences"
    unauthorised_absences: str = "unauthorised_absences"
    approved_activities: str = "approved_activities"
    total_absences: str = "total_absences"
    total_nonabsences: str = "total_nonabsences"

    data_date: str = DATA_DATE

    term_end: str = "term_end"
    term_type: str = "term_type"
    has_attendance_data: str = "has_attendance_data"


AttendanceAddedDataColumns = _AttendanceAddedDataColumns()


@dataclass
class _AttendanceDataColumns(
    _AttendanceOriginalDataColumns, _AttendanceAddedDataColumns
):
    pass


AttendanceDataColumns = _AttendanceDataColumns()

ATTENDANCE_COLUMN_RENAME = {
    "/": AttendanceOriginalDataColumns.present_am,
    "B": AttendanceOriginalDataColumns.offsite_education_approved,
    "C": AttendanceOriginalDataColumns.other_authorised,
    "D": AttendanceOriginalDataColumns.dual_registered,
    "DoB": AttendanceOriginalDataColumns.date_of_birth,
    "E": AttendanceOriginalDataColumns.excluded_authorised,
    "EnrolStatus": AttendanceOriginalDataColumns.enrol_status,
    "EntryDate": AttendanceOriginalDataColumns.entry_date,
    "Estab": AttendanceOriginalDataColumns.establishment_number,
    "Forename": AttendanceOriginalDataColumns.forename,
    "FormerSurname": AttendanceOriginalDataColumns.former_surname,
    "G": AttendanceOriginalDataColumns.holiday_unauthorised,
    "Gender": AttendanceOriginalDataColumns.gender,
    "H": AttendanceOriginalDataColumns.holiday_authorised,
    "I": AttendanceOriginalDataColumns.illness_authorised,
    "J": AttendanceOriginalDataColumns.interview_approved,
    "L": AttendanceOriginalDataColumns.late_present,
    "M": AttendanceOriginalDataColumns.medical_authorised,
    "Middlenames": AttendanceOriginalDataColumns.middlenames,
    "N": AttendanceOriginalDataColumns.unknown_unauthorised,
    "NCyearActual": AttendanceOriginalDataColumns.nc_year_actual,
    "O": AttendanceOriginalDataColumns.other_unauthorised,
    "P": AttendanceOriginalDataColumns.sports_approved,
    "PreferredSurname": AttendanceOriginalDataColumns.preferred_surname,
    "R": AttendanceOriginalDataColumns.religious_authorised,
    "S": AttendanceOriginalDataColumns.study_leave_authorised,
    "Surname": AttendanceOriginalDataColumns.surname,
    "T": AttendanceOriginalDataColumns.traveller_authorised,
    "T_Reason_C": AttendanceOriginalDataColumns.other_authorised,
    "T_Reason_E": AttendanceOriginalDataColumns.excluded_authorised,
    "T_Reason_F": AttendanceOriginalDataColumns.code_f,  # This one will be dropped since it is only in oct14 and always 0
    "T_Reason_G": AttendanceOriginalDataColumns.holiday_unauthorised,
    "T_Reason_H": AttendanceOriginalDataColumns.holiday_authorised,
    "T_Reason_I": AttendanceOriginalDataColumns.illness_authorised,
    "T_Reason_M": AttendanceOriginalDataColumns.medical_authorised,
    "T_Reason_N": AttendanceOriginalDataColumns.unknown_unauthorised,
    "T_Reason_O": AttendanceOriginalDataColumns.other_unauthorised,
    "T_Reason_R": AttendanceOriginalDataColumns.religious_authorised,
    "T_Reason_S": AttendanceOriginalDataColumns.study_leave_authorised,
    "T_Reason_T": AttendanceOriginalDataColumns.traveller_authorised,
    "T_Reason_U": AttendanceOriginalDataColumns.late_unauthorised,
    "TermlySessionsAuthorised": AttendanceOriginalDataColumns.termly_sessions_authorised,
    "TermlySessionsPossible": AttendanceOriginalDataColumns.termly_sessions_possible,
    "TermlySessionsUnauthorised": AttendanceOriginalDataColumns.termly_sessions_unauthorised,
    "U": AttendanceOriginalDataColumns.late_unauthorised,
    "UPN": AttendanceOriginalDataColumns.upn,
    "V": AttendanceOriginalDataColumns.educational_visit_approved,
    "W": AttendanceOriginalDataColumns.work_experience_approved,
    "X": AttendanceOriginalDataColumns.covid,
    "Y": AttendanceOriginalDataColumns.exceptional_circumstances,
    "\\": AttendanceOriginalDataColumns.present_pm,
}

# These are the different types of attendance datasets we can produce
@dataclass
class _AttendanceTypes:
    term_normalized: str = "term_normalized"
    percent1: str = "percent1"
    percent2: str = "percent2"
    exact: str = "exact"


AttendanceTypes = _AttendanceTypes()
