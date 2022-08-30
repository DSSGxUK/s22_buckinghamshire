"""
This file contains the constants related to the Census datasets.
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

# School Census
@dataclass
class _CensusDataOriginalColumns:
    address_line1: str = "address_line1"
    address_line2: str = "address_line2"
    address_line3: str = "address_line3"
    address_line4: str = "address_line4"
    address_line5: str = "address_line5"
    administrative_area: str = "administrative_area"
    age: str = AGE
    date_of_birth: str = "date_of_birth"
    enrol_status: str = "enrol_status"
    entry_date: str = "entry_date"
    establishment_number: str = "establishment_number"
    ethnicity: str = ETHNICITY
    forename: str = "forename"
    former_surname: str = "former_surname"
    fsme_on_census_day: str = "fsme_on_census_day"
    gender: str = GENDER
    language: str = "language"
    locality: str = "locality"
    middlenames: str = "middlenames"
    native_id: str = "native_id"
    nc_year_actual: str = NC_YEAR_ACTUAL
    paon: str = "paon"
    post_town: str = "post_town"
    postcode: str = "postcode"
    preferred_surname: str = "preferred_surname"
    resourced_provision_indicator: str = "resourced_provision_indicator"
    saon: str = "saon"
    sen_need1: str = "sen_need1"
    sen_need2: str = "sen_need2"
    sen_provision: str = "sen_provision"
    sen_unit_indicator: str = "sen_unit_indicator"
    street: str = "street"
    surname: str = "surname"
    town: str = "town"
    unique_learner_number: str = "unique_learner_number"
    upn: str = UPN


CensusDataOriginalColumns = _CensusDataOriginalColumns()

# Added columns
@dataclass
class _CensusDataAdditionalColumns:
    data_date: str = DATA_DATE

    year: str = YEAR
    census_period_end: str = "census_period_end"
    has_census_data: str = "has_census_data"
    sen_need: str = "sen_need"


CensusDataAdditionalColumns = _CensusDataAdditionalColumns()


@dataclass
class _CensusDataColumns(_CensusDataOriginalColumns, _CensusDataAdditionalColumns):
    pass


CensusDataColumns = _CensusDataColumns()

SCHOOL_CENSUS_COLUMN_RENAME = {
    "AddressLine1": CensusDataColumns.address_line1,
    "AddressLine2": CensusDataColumns.address_line2,
    "AddressLine3": CensusDataColumns.address_line3,
    "AddressLine4": CensusDataColumns.address_line4,
    "AddressLine5": CensusDataColumns.address_line5,
    "AdministrativeArea": CensusDataColumns.administrative_area,
    "Age": CensusDataColumns.age,
    "DoB": CensusDataColumns.date_of_birth,
    "EnrolStatus": CensusDataColumns.enrol_status,
    "EntryDate": CensusDataColumns.entry_date,
    "Estab": CensusDataColumns.establishment_number,
    "Ethnicity": CensusDataColumns.ethnicity,
    "FSME on Census Day": CensusDataColumns.fsme_on_census_day,
    "Forename": CensusDataColumns.forename,
    "FormerSurname": CensusDataColumns.former_surname,
    "Gender": CensusDataColumns.gender,
    "Language": CensusDataColumns.language,
    "Locality": CensusDataColumns.locality,
    "Middlenames": CensusDataColumns.middlenames,
    "NCyearActual": CensusDataColumns.nc_year_actual,
    "NativeID": CensusDataColumns.native_id,
    "PAON": CensusDataColumns.paon,
    "Postcode": CensusDataColumns.postcode,
    "PreferredSurname": CensusDataColumns.preferred_surname,
    "ResourcedProvisionIndicator": CensusDataColumns.resourced_provision_indicator,
    "SAON": CensusDataColumns.saon,
    "SENNeed1": CensusDataColumns.sen_need1,
    "SENNeed2": CensusDataColumns.sen_need2,
    "SENneed1": CensusDataColumns.sen_need1,
    "SENneed2": CensusDataColumns.sen_need2,
    "SENprovision": CensusDataColumns.sen_provision,
    "SENunitIndicator": CensusDataColumns.sen_unit_indicator,
    "Street": CensusDataColumns.street,
    "Surname": CensusDataColumns.surname,
    "Town": CensusDataColumns.town,
    "UPN": CensusDataColumns.upn,
    "UniqueLearnerNumber": CensusDataColumns.unique_learner_number,
    "postTown": CensusDataColumns.post_town,
}
