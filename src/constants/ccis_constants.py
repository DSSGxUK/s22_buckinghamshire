"""
This file contains the constants related to the CCIS datasets.
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
from dataclasses import asdict, dataclass

from .shared_constants import *
from .school_info_constants import *

@dataclass
class _CharacteristicsOriginalColumns:
    upn: str = UPN
    ethnicity: str = ETHNICITY
    address_line1: str = "address_line1"
    address_line2: str = "address_line2"
    address_line3: str = "address_line3"
    address_line4: str = "address_line4"
    family_name: str = "family_name"
    given_name: str = "given_name"
    middle_name: str = "middle_name"
    gender: str = "gender"
    la_establishment_number: str = SchoolInfoColumns.la_establishment_number
    sen_support_flag: str = "sen_support_flag"
    send_flag: str = "send_flag"
    characteristic_code: str = "characteristic_code"
    level_of_need_code: str = "level_of_need_code"
    date_of_birth: str = "date_of_birth"
    month_year_of_birth: str = "month_year_of_birth"
    postcode: str = "postcode"
    town: str = "town"
    county: str = "county"
    


CharacteristicsOriginalColumns = _CharacteristicsOriginalColumns()


@dataclass
class _CharacteristicsAddedColumns:
    data_date: str = DATA_DATE
    
    has_characteristics_data: str = "has_characteristics_data"
    age: str = AGE
    year: str = YEAR
    birth_year: str = "birth_year"
    birth_month: str = "birth_month"


CharacteristicsAddedColumns = _CharacteristicsAddedColumns()


@dataclass
class _CharacteristicsDataColumns(
    _CharacteristicsOriginalColumns, _CharacteristicsAddedColumns
):
    pass


CharacteristicsDataColumns = _CharacteristicsDataColumns()


@dataclass
class _CCISOriginalDataColumns(_CharacteristicsOriginalColumns):
    activity_code: str = "activity_code"
    cohort_status: str = "cohort_status"
    currency_lapsed: str = "currency_lapsed"
    database_id: str = "database_id"
    date_ascertained: str = "date_ascertained"
    date_of_send: str = "date_of_send"
    date_verified: str = "date_verified"
    due_to_lapse_date: str = "due_to_lapse_date"
    educated_lea: str = "educated_lea"
    establishment_name: str = SchoolInfoColumns.establishment_name
    ethnicity: str = ETHNICITY
    family_name: str = "family_name"
    gender: str = GENDER
    given_name: str = "given_name"
    guarantee_status: str = "guarantee_status"
    guarantee_status3: str = "guarantee_status3"
    guarantee_status_indicator: str = "guarantee_status_indicator"
    intended_destination_yr11: str = "intended_destination_yr11"
    la_establishment_number: str = SchoolInfoColumns.la_establishment_number
    lea_code: str = "lea_code"
    lea_code2: str = "lea_code2"
    lea_code4: str = "lea_code4"
    lea_code_at_year11: str = "lea_code_at_year11"
    lead_lea: str = "lead_lea"
    level_of_need_code: str = "level_of_need_code"
    middle_name: str = "middle_name"
    month_year_of_birth: str = "month_year_of_birth"
    neet_start_date: str = "neet_start_date"
    period_end: str = "period_end"
    postcode: str = "postcode"
    predicted_end_date: str = "predicted_end_date"
    previous_ypid_identifier: str = "previous_ypid_identifier"
    review_date: str = "review_date"
    sen_support_flag: str = "sen_support_flag"
    send_flag: str = "send_flag"
    start_date: str = "start_date"
    supplier_name: str = "supplier_name"
    supplier_xml_version: str = "supplier_xml_version"
    town: str = "town"
    transferred_to_la_code: str = "transferred_to_la_code"
    uk_provider_reference_number: str = "uk_provider_reference_number"
    unique_learner_no: str = "unique_learner_no"
    upn: str = UPN
    xml_schema_version: str = "xml_schema_version"
    young_persons_id: str = "young_persons_id"
    youth_contract_indicator: str = "youth_contract_indicator"
    youth_contract_start_date: str = "youth_contract_start_date"


CCISOriginalDataColumns = _CCISOriginalDataColumns()


@dataclass
class _CCISAddedDataColumns(_CharacteristicsAddedColumns):
    data_date: str = DATA_DATE

    # Added columns
    year: str = YEAR
    ccis_period_end: str = "ccis_period_end"
    birth_year: str = "birth_year"
    birth_month: str = "birth_month"
    neet: str = "neet"
    compulsory_school: str = "compulsory_school"
    unknown: str = "unknown"
    unknown_currently: str = "unknown_currently"
    compulsory_school_always: str = "compulsory_school_always"

    has_ccis_data: str = "has_ccis_data"


CCISAddedDataColumns = _CCISAddedDataColumns()

# These are the different target variables for training the model
@dataclass
class _Targets:
    # Target columns
    neet_ever: str = "neet_ever"
    unknown_ever: str = "unknown_ever"
    neet_unknown_ever: str = "neet_unknown_ever"


Targets = _Targets()


@dataclass
class _CCISDataColumns(_CCISOriginalDataColumns, _CCISAddedDataColumns, _Targets):
    pass


CCISDataColumns = _CCISDataColumns()

non_prediction_columns = list(asdict(Targets).values()) + [
    CCISDataColumns.compulsory_school_always,
    CCISDataColumns.unknown_currently,
]

CCIS_COLUMN_RENAME = {
    "ActivityCode": CCISOriginalDataColumns.activity_code,
    "AddressLine1": CCISOriginalDataColumns.address_line1,
    "AddressLine2": CCISOriginalDataColumns.address_line2,
    "AddressLine3": CCISOriginalDataColumns.address_line3,
    "AddressLine4": CCISOriginalDataColumns.address_line4,
    "CharacteristicCode": CCISOriginalDataColumns.characteristic_code,
    "CohortStatus": CCISOriginalDataColumns.cohort_status,
    "County": CCISOriginalDataColumns.county,
    "CurrencyLapsed": CCISOriginalDataColumns.currency_lapsed,
    "DatabaseId": CCISOriginalDataColumns.database_id,
    "DateAscertained": CCISOriginalDataColumns.date_ascertained,
    "DateOfSend": CCISOriginalDataColumns.date_of_send,
    "DateVerified": CCISOriginalDataColumns.date_verified,
    "DOB": CCISOriginalDataColumns.date_of_birth,
    "DueToLapseDate": CCISOriginalDataColumns.due_to_lapse_date,
    "EducatedLEA": CCISOriginalDataColumns.educated_lea,
    "Estab": CCISOriginalDataColumns.la_establishment_number,
    "EstablishmentName": CCISOriginalDataColumns.establishment_name,
    "EstablishmentNumber": CCISOriginalDataColumns.la_establishment_number,
    "Ethnicity": CCISOriginalDataColumns.ethnicity,
    "FamilyName": CCISOriginalDataColumns.family_name,
    "Gender": CCISOriginalDataColumns.gender,
    "GivenName": CCISOriginalDataColumns.given_name,
    "GuaranteeStatus": CCISOriginalDataColumns.guarantee_status,
    "GuaranteeStatus3": CCISOriginalDataColumns.guarantee_status3,
    "GuaranteeStatusIndicator": CCISOriginalDataColumns.guarantee_status_indicator,
    "IntendedDestinationYr11": CCISOriginalDataColumns.intended_destination_yr11,
    "LEACode": CCISOriginalDataColumns.lea_code,
    "LEACode2": CCISOriginalDataColumns.lea_code2,
    "LEACode4": CCISOriginalDataColumns.lea_code4,
    "LEACodeAtYear11": CCISOriginalDataColumns.lea_code_at_year11,
    "LeadLEA": CCISOriginalDataColumns.lead_lea,
    "LevelOfNeedCode": CCISOriginalDataColumns.level_of_need_code,
    "LevelofNeedCode": CCISOriginalDataColumns.level_of_need_code,
    "MiddleName": CCISOriginalDataColumns.middle_name,
    "NEETStartDate": CCISOriginalDataColumns.neet_start_date,
    "PeriodEnd": CCISOriginalDataColumns.period_end,
    "Postcode": CCISOriginalDataColumns.postcode,
    "PredictedEndDate": CCISOriginalDataColumns.predicted_end_date,
    "PreviousYPIDIdentifier": CCISOriginalDataColumns.previous_ypid_identifier,
    "ReviewDate": CCISOriginalDataColumns.review_date,
    "SENDFlag": CCISOriginalDataColumns.send_flag,
    "SENSupportFlag": CCISOriginalDataColumns.sen_support_flag,
    "StartDate": CCISOriginalDataColumns.start_date,
    "SupplierName": CCISOriginalDataColumns.supplier_name,
    "SupplierXMLVersion": CCISOriginalDataColumns.supplier_xml_version,
    "Town": CCISOriginalDataColumns.town,
    "TransferredToLACode": CCISOriginalDataColumns.transferred_to_la_code,
    "UKProviderReferenceNumber": CCISOriginalDataColumns.uk_provider_reference_number,
    "UPN": CCISOriginalDataColumns.upn,
    "UniqueLearnerNo": CCISOriginalDataColumns.unique_learner_no,
    "UniquePupilNumber": CCISOriginalDataColumns.upn,
    "XMLSchemaVersion": CCISOriginalDataColumns.xml_schema_version,
    "YoungPersonsID": CCISOriginalDataColumns.young_persons_id,
    "YouthContractIndicator": CCISOriginalDataColumns.youth_contract_indicator,
    "YouthContractStartDate": CCISOriginalDataColumns.youth_contract_start_date,
    "month/year of birth": CCISOriginalDataColumns.month_year_of_birth,
    "moth/year of birth": CCISOriginalDataColumns.month_year_of_birth,
}
