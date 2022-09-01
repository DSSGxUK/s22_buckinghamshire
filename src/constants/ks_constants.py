"""
This file contains the constants related to the Key Stage datasets.
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

# This is used to suffix columns with both numerical values
# and codes.
CODED_SUFFIX = "_codes"

# This code is used for students who took a ks2 test
# and received a grade.
TEST_TAKEN_CODE = "test_taken"

# KS4 Data
@dataclass
class _PupilDeprivationColumns:
    """
    These are columns used to identify deprivation and disadvantaged
    statusus for students from the key stage 4 dataset. This class
    also helps specify what key stage 2 data we would like from the
    partner before the student starts year 11.
    """

    upn: str = UPN
    sen: str = "sen"
    deprivation_indicator_idaci_score: str = "deprivation_indicator_idaci_score"
    is_the_pupil_disadvantaged: str = "is_the_pupil_disadvantaged"
    is_the_pupil_looked_after: str = "is_the_pupil_looked_after"


PupilDeprivationColumns = _PupilDeprivationColumns()
@dataclass
class _KS2OriginalColumns(_PupilDeprivationColumns):
    """
    These are columns used to identify key stage 2 data from
    the key stage 4 dataset. This class also helps specify what
    key stage 2 data we would like from the partner before the student
    starts year 11.
    """

    upn: str = UPN
    ks2_english: str = "ks2_english"
    ks2_mathematics: str = "ks2_mathematics"
    ks2_aps_va: str = "ks2_aps_va"
    ks2_aps_cubed: str = "ks2_aps_cubed"
    ks2_aps_squared: str = "ks2_aps_squared"
    ks2_english_ta: str = "ks2_english_ta"
    ks2_english_ps: str = "ks2_english_ps"
    ks2_english_ps_deviation: str = "ks2_english_ps_deviation"
    ks2_english_ta_level: str = "ks2_english_ta_level"
    ks2_english_level_finely_graded: str = "ks2_english_level_finely_graded"
    ks2_mathematics_ta: str = "ks2_mathematics_ta"
    ks2_mathematics_ta_level: str = "ks2_mathematics_ta_level"
    ks2_mathematics_level_finely_graded: str = "ks2_mathematics_level_finely_graded"
    ks2_maths_ps: str = "ks2_maths_ps"
    ks2_maths_ps_deviation: str = "ks2_maths_ps_deviation"
    ks2_prior_band: str = "ks2_prior_band"
    ks2_reading_ps: str = "ks2_reading_ps"
    ks2_reading_ta_level: str = "ks2_reading_ta_level"
    ks2_reading_level_finely_graded: str = "ks2_reading_level_finely_graded"
    ks2_mathematics_finely_graded: str = "ks2_mathematics_finely_graded"
    la_establishment_number: str = SchoolInfoColumns.la_establishment_number


KS2OriginalColumns = _KS2OriginalColumns()


@dataclass
class _KS2AddedColumns:
    has_ks2_data: str = "has_ks2_data"

KS2AddedColumns = _KS2AddedColumns()

@dataclass 
class _KS2Columns(_KS2OriginalColumns, _KS2AddedColumns):
    pass

KS2Columns = _KS2Columns()

@dataclass
class _KS4Columns:
    upn: str = UPN
    attainment_8_score: str = "attainment_8_score"


KS4Columns = _KS4Columns()


@dataclass
class _KSAddedColumns:
    data_date: str = DATA_DATE

    ks4_period_end: str = "ks4_period_end"
    has_ks2_data: str = "has_ks2_data"


KSAddedColumns = _KSAddedColumns()

# If we need to, we can add KS4 columns too.
@dataclass
class _KSDataColumns(
    _KS2Columns, _PupilDeprivationColumns, _KS4Columns, _KSAddedColumns
):
    pass


KSDataColumns = _KSDataColumns()

# Normally all the values on the right-hand side that
# we are renaming old columns to would be specified in a dataclass
# first, then referenced below. However, there are a lot of KS columns,
# so to avoid cluttering this file, we only pulled out the columns
# that were useful to us.
KS4_COLUMN_RENAME = {
    "'EBacc English' VA score": "ebacc_english_va_score",
    "'EBacc Humanities' VA score": "ebacc_humanities_va_score",
    "'EBacc Language' VA score": "ebacc_language_va_score",
    "'EBacc Maths' VA score": "ebacc_maths_va_score",
    "'EBacc Science' VA score": "ebacc_science_va_score",
    "1/2 Cert English Language": "1/2_cert_english_language",
    "1/2 Cert English Literature": "1/2_cert_english_literature",
    "5+ A*-C inc EM - GCSEs only": "5_plus_a_star_c_inc_em_gcses_only",
    "AFSM Additional Math FSM": "afsm_additional_math_fsm",
    "AGE": "age",
    "Achieved English EBacc": "achieved_english_ebacc",
    "Achieved English EBacc 9-4": "achieved_english_ebacc_9_4",
    "Achieved English EBacc 9-5": "achieved_english_ebacc_9_5",
    "Achieved Humanities EBacc": "achieved_humanities_ebacc",
    "Achieved Humanities EBacc at grade 9-4": "achieved_humanities_ebacc_at_grade_9_4",
    "Achieved Humanities EBacc at grade 9-5": "achieved_humanities_ebacc_at_grade_9_5",
    "Achieved Language EBacc": "achieved_language_ebacc",
    "Achieved Language EBacc at grade 9-4": "achieved_language_ebacc_at_grade_9_4",
    "Achieved Language EBacc at grade 9-5": "achieved_language_ebacc_at_grade_9_5",
    "Achieved Maths EBacc": "achieved_maths_ebacc",
    "Achieved Maths EBacc 9-4": "achieved_maths_ebacc_9_4",
    "Achieved Maths EBacc 9-5": "achieved_maths_ebacc_9_5",
    "Achieved Science EBacc": "achieved_science_ebacc",
    "Achieved Science EBacc at grade 9-4": "achieved_science_ebacc_at_grade_9_4",
    "Achieved Science EBacc at grade 9-5": "achieved_science_ebacc_at_grade_9_5",
    "Attainment 8 score": "attainment_8_score",
    "BCert1 Building": "bcert1_building",
    "BCert1 Multimedia": "bcert1_multimedia",
    "BTC1/2C Business Studies": "btc1/2c_business_studies",
    "BTC1/2C Health Studies": "btc1/2c_health_studies",
    "BTC1/2C Multimedia": "btc1/2c_multimedia",
    "BTC1/2C Sports Studies": "btc1/2c_sports_studies",
    "BTC1/2C Travel & Tourism": "btc1/2c_travel_and_tourism",
    "BTC1/2EC Computer Systems": "btc1/2ec_computer_systems",
    "BTC1Awd Applied Sciences": "btc1awd_applied_sciences",
    "BTC1Awd Art & Design": "btc1awd_art_and_design",
    "BTC1Awd Building": "btc1awd_building",
    "BTC1Awd Business Studies": "btc1awd_business_studies",
    "BTC1Awd Childcare Skills": "btc1awd_childcare_skills",
    "BTC1Awd Computer Systems": "btc1awd_computer_systems",
    "BTC1Awd Engineering Studies": "btc1awd_engineering_studies",
    "BTC1Awd Health Studies": "btc1awd_health_studies",
    "BTC1Awd Multimedia": "btc1awd_multimedia",
    "BTC1Awd Music Studies": "btc1awd_music_studies",
    "BTC1Awd Performing Arts": "btc1awd_performing_arts",
    "BTC1Awd Sports Studies": "btc1awd_sports_studies",
    "BTC1Awd Travel & Tourism": "btc1awd_travel_and_tourism",
    "Best 8 VA score": "best_8_va_score",
    "CamNatCer Computer Use": "camnatcer_computer_use",
    "CamNatCer Small Business": "camnatcer_small_business",
    "Candidate Number": "candidate_number",
    "Candidate number": "candidate_number",
    "Capped GCSE point score": "capped_gcse_point_score",
    "Capped Point Score": "capped_point_score",
    "Capped point score (VA)": "capped_point_score_va",
    "Capped score + bonuses": "capped_score_+_bonuses",
    "Date of birth": "date_of_birth",
    "Deprivation indicator - IDACI score": PupilDeprivationColumns.deprivation_indicator_idaci_score,
    "DfE Establishment Number": KS2Columns.la_establishment_number,
    "Did the pupil join within the last 2 yrs?": "did_the_pupil_join_within_the_last_2_yrs",
    "EAL Group": "eal_group",
    "EBacc Humanities' VA score": "ebacc_humanities_va_score",
    "EBacc Language' VA score": "ebacc_language_va_score",
    "EBacc Science' VA score": "ebacc_science_va_score",
    "EBacc slots filled": "ebacc_slots_filled",
    "Edex D Cert Science Double Awd": "edex_d_cert_science_double_awd",
    "Edexcel Cert Biology": "edexcel_cert_biology",
    "Edexcel Cert Chemistry": "edexcel_cert_chemistry",
    "Edexcel Cert English Language": "edexcel_cert_english_language",
    "Edexcel Cert English Literature": "edexcel_cert_english_literature",
    "Edexcel Cert Mathematics": "edexcel_cert_mathematics",
    "Edexcel Cert Physics": "edexcel_cert_physics",
    "English & maths GCSEs A*-C?": "took_english_and_maths_gcses_a*_c",
    "English Bacc 9-4?": "took_english_bacc_9_4",
    "English Bacc 9-5?": "took_english_bacc_9_5",
    "English Bacc?": "took_english_bacc",
    "English Bonus": "english_bonus",
    "English EBacc PS (VA)": "english_ebacc_ps_va",
    "English and maths GCSEs 9-4?": "took_english_and_maths_gcses_9_4",
    "English and maths GCSEs 9-5?": "took_english_and_maths_gcses_9_5",
    "English exclude previous school": "english_exclude_previous_school",
    "Entered all three core science pathway subjects": "entered_all_three_core_science_pathway_subjects",
    "Entries": "entries",
    "Entry ALL EBacc": "entry_all_ebacc",
    "Entry English EBacc": "entry_english_ebacc",
    "Entry Humanities EBacc": "entry_humanities_ebacc",
    "Entry Language EBacc": "entry_language_ebacc",
    "Entry Maths EBacc": "entry_maths_ebacc",
    "Entry Science EBacc": "entry_science_ebacc",
    "Entry multiple languages": "entry_multiple_languages",
    "Entry triple science": "entry_triple_science",
    "Expected Progress  Maths KS2-4": "expected_progress_maths_ks2_4",
    "Expected Progress English KS2-4": "expected_progress_english_ks2_4",
    "Forename": "forename",
    "Forvus index number": "forvus_index_number",
    "GCE AS Chinese": "gce_as_chinese",
    "GCE AS Critical Thinking": "gce_as_critical_thinking",
    "GCE AS Drama & Theat.Stds": "gce_as_drama_and_theat_stds",
    "GCE AS French": "gce_as_french",
    "GCE AS General Studies": "gce_as_general_studies",
    "GCE AS German": "gce_as_german",
    "GCE AS Italian": "gce_as_italian",
    "GCE AS Mathematics": "gce_as_mathematics",
    "GCE AS Polish": "gce_as_polish",
    "GCE AS Punjabi": "gce_as_punjabi",
    "GCE AS Religious Studies": "gce_as_religious_studies",
    "GCE AS Russian": "gce_as_russian",
    "GCE AS World Development": "gce_as_world_development",
    "GCSE Ancient History": "gcse_ancient_history",
    "GCSE Application of Math": "gcse_application_of_math",
    "GCSE Applied Art & Des": "gcse_applied_art_and_des",
    "GCSE Applied Engineering": "gcse_applied_engineering",
    "GCSE Arabic": "gcse_arabic",
    "GCSE Art & Des(Graphcs)": "gcse_art_and_desgraphcs",
    "GCSE Art & Des(Photo.)": "gcse_art_and_desphoto_",
    "GCSE Art & Des(Textles)": "gcse_art_and_destextles",
    "GCSE Art & Design": "gcse_art_and_design",
    "GCSE Art&Des : Fine Art": "gcse_artanddes_fine_art",
    "GCSE Biology": "gcse_biology",
    "GCSE Bus. Studs:Single": "gcse_bus_studs_single",
    "GCSE Chemistry": "gcse_chemistry",
    "GCSE Chinese": "gcse_chinese",
    "GCSE Class.Civilisation": "gcse_class_civilisation",
    "GCSE Com.Stds/Computing": "gcse_com_stds/computing",
    "GCSE D&T Electrnc.Prods": "gcse_dandt_electrnc_prods",
    "GCSE D&T Engineering": "gcse_dandt_engineering",
    "GCSE D&T Food Technolgy": "gcse_dandt_food_technolgy",
    "GCSE D&T Graphic Prods": "gcse_dandt_graphic_prods",
    "GCSE D&T Product Design": "gcse_dandt_product_design",
    "GCSE D&T Resist. Matrls": "gcse_dandt_resist_matrls",
    "GCSE D&T Textiles Tech.": "gcse_dandt_textiles_tech_",
    "GCSE Dance": "gcse_dance",
    "GCSE Drama & Theat.Stds": "gcse_drama_and_theat_stds",
    "GCSE Dutch": "gcse_dutch",
    "GCSE Economics": "gcse_economics",
    "GCSE English": "gcse_english",
    "GCSE English Language": "gcse_english_language",
    "GCSE English Literature": "gcse_english_literature",
    "GCSE Entries": "gcse_entries",
    "GCSE Film Studies": "gcse_film_studies",
    "GCSE French": "gcse_french",
    "GCSE General Studies": "gcse_general_studies",
    "GCSE Geography": "gcse_geography",
    "GCSE German": "gcse_german",
    "GCSE Gujarati": "gcse_gujarati",
    "GCSE HE: Child Devt": "gcse_he_child_devt",
    "GCSE HE: Food": "gcse_he_food",
    "GCSE Health & Soc Care": "gcse_health_and_soc_care",
    "GCSE History": "gcse_history",
    "GCSE Humanities: Single": "gcse_humanities_single",
    "GCSE Inform Comm Tech": "gcse_inform_comm_tech",
    "GCSE Italian": "gcse_italian",
    "GCSE Japanese": "gcse_japanese",
    "GCSE Latin": "gcse_latin",
    "GCSE Law": "gcse_law",
    "GCSE Mathematics": "gcse_mathematics",
    "GCSE Media/Film/TV Stds": "gcse_media/film/tv_stds",
    "GCSE Method in Math": "gcse_method_in_math",
    "GCSE Modern Greek": "gcse_modern_greek",
    "GCSE Music": "gcse_music",
    "GCSE Office Technology": "gcse_office_technology",
    "GCSE Performing Arts": "gcse_performing_arts",
    "GCSE Physics": "gcse_physics",
    "GCSE Polish": "gcse_polish",
    "GCSE Portuguese": "gcse_portuguese",
    "GCSE Psychology": "gcse_psychology",
    "GCSE Religious Studies": "gcse_religious_studies",
    "GCSE Russian": "gcse_russian",
    "GCSE Sci: Electronics": "gcse_sci_electronics",
    "GCSE Science (Core)": "gcse_science_core",
    "GCSE Science: Additional": "gcse_science_additional",
    "GCSE Science: Astronomy": "gcse_science_astronomy",
    "GCSE Science: Geology": "gcse_science_geology",
    "GCSE Soc Sci:Citizenshp": "gcse_soc_sci_citizenshp",
    "GCSE Sociology": "gcse_sociology",
    "GCSE Spanish": "gcse_spanish",
    "GCSE Sport/P.E. Studies": "gcse_sport/p_e_studies",
    "GCSE Statistics": "gcse_statistics",
    "GCSE Urdu": "gcse_urdu",
    "Gender": "gender",
    "GradeEx Music Perf (Group)": "gradeex_music_perf_group",
    "Has the pupil been adopted from care?": "has_the_pupil_been_adopted_from_care",
    "Has the pupil been eligible for FSM in the last 6 years?": "has_the_pupil_been_eligible_for_fsm_in_the_last_6_years",
    "Highest point score achieved in humanities Ebacc subject area": "highest_point_score_achieved_in_humanities_ebacc_subject_area",
    "Highest point score achieved in language Ebacc subject area": "highest_point_score_achieved_in_language_ebacc_subject_area",
    "Humanities EBacc PS (VA)": "humanities_ebacc_ps_va",
    "IGCSE CIC Biology": "igcse_cic_biology",
    "IGCSE CIC Bus. Studs:Single": "igcse_cic_bus_studs_single",
    "IGCSE CIC Chemistry": "igcse_cic_chemistry",
    "IGCSE CIC English Language": "igcse_cic_english_language",
    "IGCSE CIC English Literature": "igcse_cic_english_literature",
    "IGCSE CIC German": "igcse_cic_german",
    "IGCSE CIC History": "igcse_cic_history",
    "IGCSE CIC Inform Comm Tech": "igcse_cic_inform_comm_tech",
    "IGCSE CIC Physics": "igcse_cic_physics",
    "Is the pupil disadvantaged?": PupilDeprivationColumns.is_the_pupil_disadvantaged,
    "Is the pupil looked after?": PupilDeprivationColumns.is_the_pupil_looked_after,
    "KS2  ENGLISH (KS2)": KS2Columns.ks2_english,
    "KS2  MATHEMATICS (KS2)": KS2Columns.ks2_mathematics,
    "KS2 APS (VA)": KS2Columns.ks2_aps_va,
    "KS2 APS cubed": KS2Columns.ks2_aps_cubed,
    "KS2 APS squared": KS2Columns.ks2_aps_squared,
    "KS2 ENGLISH TA": KS2Columns.ks2_english_ta,
    "KS2 English PS": KS2Columns.ks2_english_ps,
    "KS2 English PS deviation": KS2Columns.ks2_english_ps_deviation,
    "KS2 English TA level": KS2Columns.ks2_english_ta_level,
    "KS2 English level (finely graded)": KS2Columns.ks2_english_level_finely_graded,
    "KS2 MATHEMATICS TA": KS2Columns.ks2_mathematics_ta,
    "KS2 Mathematics TA level": KS2Columns.ks2_mathematics_ta_level,
    "KS2 Mathematics level (finely graded)": KS2Columns.ks2_mathematics_level_finely_graded,
    "KS2 Maths PS": KS2Columns.ks2_maths_ps,
    "KS2 Maths PS deviation": KS2Columns.ks2_maths_ps_deviation,
    "KS2 Prior Band": KS2Columns.ks2_prior_band,
    "KS2 Reading PS": KS2Columns.ks2_reading_ps,
    "KS2 Reading TA level": KS2Columns.ks2_reading_ta_level,
    "KS2 Reading level (finely graded)": KS2Columns.ks2_reading_level_finely_graded,
    "KS2 mathematics (finely graded)": KS2Columns.ks2_mathematics_finely_graded,
    "KS2-4 predicted PS for 'Best 8' VA": "ks2_4_predicted_ps_for_best_8_va",
    "KS2-4 predicted PS for 'EBacc English' VA": "ks2_4_predicted_ps_for_ebacc_english_va",
    "KS2-4 predicted PS for 'EBacc Humanities' VA": "ks2_4_predicted_ps_for_ebacc_humanities_va",
    "KS2-4 predicted PS for 'EBacc Language' VA": "ks2_4_predicted_ps_for_ebacc_language_va",
    "KS2-4 predicted PS for 'EBacc Maths' VA": "ks2_4_predicted_ps_for_ebacc_maths_va",
    "KS2-4 predicted PS for 'EBacc Science' VA": "ks2_4_predicted_ps_for_ebacc_science_va",
    "Language EBacc PS (VA)": "language_ebacc_ps_va",
    "Maths Bonus": "maths_bonus",
    "Maths EBacc PS (VA)": "maths_ebacc_ps_va",
    "Maths exclude previous school": "maths_exclude_previous_school",
    "OTHGen1 Latin": "othgen1_latin",
    "OTHGen1 Untranslated lit": "othgen1_untranslated_lit",
    "OTHGen2 Additional Maths": "othgen2_additional_maths",
    "OTHGen2 Latin": "othgen2_latin",
    "Open slots filled": "open_slots_filled",
    "Performance Category": "performance_category",
    "Point score in English EBacc subject area": "point_score_in_english_ebacc_subject_area",
    "Point score in maths EBacc subject area": "point_score_in_maths_ebacc_subject_area",
    "Point score in science Ebacc subject area": "point_score_in_science_ebacc_subject_area",
    "Post Looked After Arrangements": "post_looked_after_arrangements",
    "PrLearn2 Engineering": "prlearn2_engineering",
    "PrLearn2 Hair/Personal Care": "prlearn2_hair_personal_care",
    "Previous Maths pathway flag": "previous_maths_pathway_flag",
    "Previous Pupil level English pathway flag": "previous_pupil_level_english_pathway_flag",
    "Previous Pupil level Science pathway flag": "previous_pupil_level_science_pathway_flag",
    "Prior attainment score in English and mathematics": "prior_attainment_score_in_english_and_mathematics",
    "Progress 8 score for Ebacc slots": "progress_8_score_for_ebacc_slots",
    "Progress 8 score for English": "progress_8_score_for_english",
    "Progress 8 score for mathematics": "progress_8_score_for_mathematics",
    "Progress 8 score for open slots": "progress_8_score_for_open_slots",
    "Pupil Inclusion Status Flag": "pupil_inclusion_status_flag",
    "Pupil included in progress 8 calculations": "pupil_included_in_progress_8_calculations",
    "Pupil level EBacc average point score": "pupil_level_ebacc_average_point_score",
    "Pupil level English pathway flag": "pupil_level_english_pathway_flag",
    "Pupil level Maths pathway flag": "pupil_level_maths_pathway_flag",
    "Pupil level Science pathway flag": "pupil_level_science_pathway_flag",
    "Pupil progress 8 score": "pupil_progress_8_score",
    "Pupil's Progress 8 score has been adjusted due to extreme negative score": "pupils_progress_8_score_has_been_adjusted_due_to_extreme_negative_score",
    "Pupil's adjusted progress 8 score": "pupils_adjusted_progress_8_score",
    "Pupil's unadjusted progress 8 score": "pupils_unadjusted_progress_8_score",
    "SEN": PupilDeprivationColumns.sen,
    "School URN": "school_urn",
    "Science EBacc PS (VA)": "science_ebacc_ps_va",
    "Science Pathway 1 before cut off": "science_pathway_1_before_cut_off",
    "Science Pathway 2 before cut off": "science_pathway_2_before_cut_off",
    "Science Pathway 3 before cut off": "science_pathway_3_before_cut_off",
    "Science exclude previous school": "science_exclude_previous_school",
    "Science first entry pathway": "science_first_entry_pathway",
    "Score achieved in 1st EBacc slot": "score_achieved_in_1st_ebacc_slot",
    "Score achieved in 1st open slot": "score_achieved_in_1st_open_slot",
    "Score achieved in 2nd EBacc slot": "score_achieved_in_2nd_ebacc_slot",
    "Score achieved in 2nd open slot": "score_achieved_in_2nd_open_slot",
    "Score achieved in 3rd EBacc slot": "score_achieved_in_3rd_ebacc_slot",
    "Score achieved in 3rd open slot": "score_achieved_in_3rd_open_slot",
    "Score achieved in Ebacc slots": "score_achieved_in_ebacc_slots",
    "Score achieved in English": "score_achieved_in_english",
    "Score achieved in mathematics": "score_achieved_in_mathematics",
    "Score achieved in open slots": "score_achieved_in_open_slots",
    "Student's ethnicity": "students_ethnicity",
    "Surname": "surname",
    "TA used for English prior attainment due to 2010 KS2 test boycott": "ta_used_for_english_prior_attainment_due_to_2010_ks2_test_boycott",
    "TA used for maths prior attainment due to 2010 KS2 test boycott": "ta_used_for_maths_prior_attainment_due_to_2010_ks2_test_boycott",
    "Total GCSE point score excluding equivalents": "total_gcse_point_score_excluding_equivalents",
    "Total Point Score": "total_point_score",
    "Total score achieved in open slots - GCSEs only": "total_score_achieved_in_open_slots_gcses_only",
    "Total score achieved in open slots - non-GCSEs only": "total_score_achieved_in_open_slots_non_gcses_only",
    "UPN": "upn",
    "Use of Math Use of Maths": "use_of_math_use_of_maths",
    "VRQ2 Computer Use": "vrq2_computer_use",
    "vGCSE Applied Business": "vgcse_applied_business",
    "vGCSE Catering Studies": "vgcse_catering_studies",
    "vGCSE DA Applied Engineering": "vgcse_da_applied_engineering",
    "vGCSE DA Health & Soc Care": "vgcse_da_health_and_soc_care",
    "vGCSE Health & Soc Care": "vgcse_health_and_soc_care",
    "vGCSE Leisure & Tourism": "vgcse_leisure_and_tourism",
}
