"""
This stage is a *premerge* stage. This means it takes its input data
and prepares it for the `merge_multi_upn` stage that creates
a modeling dataset. In this case, the data is the CCIS data (aka NEET data).

This stage will perform 
- a drop of columns that cannot be used for modeling because
they are only available after the student is of age >= 15. (e.g.
`intended_yr11_destination` or `activity_code`).
- some type conversion of columns that should be numeric.
- aggregation by UPN.

Parameters
----------
input : str
    The filepath to the input CCIS annotated csv. This is usually
    `../data/interim/neet_annotated.csv`.
output : str
    The filepath to the output CCIS premerge csv. This is usually
    `../data/interim/neet_premerge.csv`
pre_age15 : bool
    This flag tells us whether we should drop columns that contain information
    about a student we can only learn after their school age is 15.

Other Parameters
----------------
only_seldom_used_keyword : int, optional
    Infrequently used parameters can be described under this optional
    section to prevent cluttering the Parameters section.
**kwargs : dict
    Other infrequently used keyword arguments. Note that all keyword
    arguments appearing after the first parameter specified under the
    Other Parameters section, should also be described under this
    section.

Validation
------
NotImplemtedError
    If `--pre_age15` is False.

"""

import argparse
import pandas as pd

# DVC Params
from src.constants import (
    NEET_PREMERGE_AGG,
    CCISDataColumns,
    SchoolInfoColumns,
    NEET_PREMERGE_AGG,
    NA_VALS,
)

# Other code
from src import data_utils as d
from src import file_utils as f
from src import log_utils as l
from src import aggregation_utils as au


def annotated_neet_data_validation(df):
    """
    TODO: Any validation for incoming data can be done here. 
    You should write your validation in terms of `assert`
    statements or raise an exception if any check fails.
    """
    pass

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--input', required=True,
                    help='where to find the input CCIS annotated csv')
parser.add_argument('--output', required=True,
                    help='where to put the output CCIS premerge csv')
parser.add_argument('--pre_age15', action='store_true',
                    help='whether or not to drop columns that are not available for students under 15 years of age')
    
if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    if not args.pre_age15:
        raise NotImplementedError("TODO: Implement keeping of students 15 or above")
    
    df = d.load_csv(  # Modeling dataset so we drop unnecessary columns and entries
        args.input, 
        drop_empty=True, 
        drop_single_valued=True,
        drop_duplicates=True,
        read_as_str=True,  # This will ensure values are not cast to floats
        use_na=True,  # and this will ensure empty values are read as nan
        drop_missing_upns=True,
        na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger
    )
    
    logger.info(f'Initial row count {len(df)}')
    logger.info(f'Initial column count {len(df.columns)}')
    
    # Do some validation
    logger.info('Validating incoming CCIS annotated data')
    annotated_neet_data_validation(df)
    
    general_drop_columns = [
        CCISDataColumns.database_id,
        CCISDataColumns.date_of_send,
        CCISDataColumns.period_end,
        CCISDataColumns.supplier_name,
        CCISDataColumns.supplier_xml_version,
        CCISDataColumns.xml_schema_version,
        CCISDataColumns.young_persons_id,
        CCISDataColumns.educated_lea,
        CCISDataColumns.lead_lea,
        CCISDataColumns.lea_code_at_year11,
        CCISDataColumns.youth_contract_start_date,
        CCISDataColumns.transferred_to_la_code,
        CCISDataColumns.start_date,
        CCISDataColumns.date_verified,
        CCISDataColumns.date_ascertained,
        CCISDataColumns.review_date,
        CCISDataColumns.due_to_lapse_date,
        SchoolInfoColumns.la_establishment_number,
        CCISDataColumns.neet_start_date,
        CCISDataColumns.predicted_end_date,
        CCISDataColumns.ccis_period_end,
        SchoolInfoColumns.establishment_number,
        SchoolInfoColumns.establishment_name,
        SchoolInfoColumns.establishment_area,
        SchoolInfoColumns.establishment_postcode,
        SchoolInfoColumns.establishment_electoral_wards,
    ]
    logger.info(f'Dropping columns {general_drop_columns}')
    df.drop(general_drop_columns, axis=1, inplace=True)
    
    # Drop data we wouldn't have for students <15 yrs old
    if args.pre_age15:
        pre_age15_drop_columns = [
            CCISDataColumns.intended_destination_yr11,
            SchoolInfoColumns.establishment_type,
            SchoolInfoColumns.establishment_status,
            CCISDataColumns.activity_code,  # Additional categories will be useful to identify unknowns
            CCISDataColumns.youth_contract_indicator,  # Null values so do categorical
            CCISDataColumns.cohort_status,  # This could be useful for unknown classification
            CCISDataColumns.guarantee_status_indicator, # This could be useful for unknown classification
            CCISDataColumns.age,
            CCISDataColumns.currency_lapsed,
            CCISDataColumns.neet,
            CCISDataColumns.unknown
        ]
        logger.info(f'Dropping columns {pre_age15_drop_columns} for pre-age15 dataset')
        df.drop(pre_age15_drop_columns, axis=1, inplace=True)
    
    
    # Create categorical column for characteristic codes and level_of_need
    df = d.get_dummies_with_logging(df, columns=[CCISDataColumns.characteristic_code, CCISDataColumns.level_of_need_code], logger=logger)
    
    logger.info('Convert columns to numeric to prep for aggregation')
    numeric_cols = [
        CCISDataColumns.sen_support_flag,
        CCISDataColumns.send_flag,
        CCISDataColumns.neet_ever,
        CCISDataColumns.unknown_ever,
        CCISDataColumns.compulsory_school,
        CCISDataColumns.birth_year,
        CCISDataColumns.birth_month,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    new_column_types = {col:  df[col].dtype for col in numeric_cols}
    logger.info(f'New column types are {new_column_types}')
    
    # Aggregate by upn
    aggregation_dict = au.build_agg_dict(NEET_PREMERGE_AGG, columns=df.columns, logger=logger)
    df = au.gby_agg_with_logging(
        df, 
        groupby_column=CCISDataColumns.upn,
        agg_dict=aggregation_dict, 
        logger=logger
    )
    logger.info(f'{len(df)} rows after aggregation. This is the number of unique upns')
    
    # Drop students who have no data after compulsory school (the compulsory_school col will be 1)
    logger.info('Dropping students who have no data after compulsory school')
    prev_len = len(df)
    df = df.loc[~(df[CCISDataColumns.compulsory_school].astype(bool))]
    logger.info(f'Dropped {prev_len - len(df)} rows.')

    df.drop(CCISDataColumns.compulsory_school, axis=1, inplace=True)
    
    logger.info(f'Final row count {len(df)}')
    logger.info(f'Final column count {len(df.columns)}')
    
    csv_fp = f.tmp_path(args.output, debug=args.debug)
    logger.info(f'Saving categorical data to {csv_fp}')
    df.to_csv(csv_fp, index=False)
    
    