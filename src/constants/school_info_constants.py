"""
This file contains the constants related to the Secondary School Info datasets.
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

# School Info
@dataclass
class _SchoolInfoColumns:
    la_establishment_number: str = "la_establishment_number"
    establishment_number: str = "establishment_number"
    establishment_name: str = "establishment_name"
    establishment_type: str = "establishment_type"
    establishment_status: str = "establishment_status"
    establishment_area: str = "establishment_area"
    establishment_postcode: str = "establishment_postcode"
    establishment_electoral_wards: str = "establishment_electoral_wards"


SchoolInfoColumns = _SchoolInfoColumns()


SCHOOL_INFO_RENAME = {
    "LAEstab (anon)": SchoolInfoColumns.la_establishment_number,
    "Estab (Anon)": SchoolInfoColumns.establishment_number,
    "School Name": SchoolInfoColumns.establishment_name,
    "Type": SchoolInfoColumns.establishment_type,
    "Status": SchoolInfoColumns.establishment_status,
    "Area": SchoolInfoColumns.establishment_area,
    "Postcode": SchoolInfoColumns.establishment_postcode,
    "Electoral Ward*": SchoolInfoColumns.establishment_electoral_wards,
}
