"""
Merges 4 datasets (NCCIS, Attendance, Census & KS2) into 1 dataset with multiple years of data per student (multi-upn dataset).
UPNs in attendance, census and KS grades are matched to the NCCIS data to join (left join to the NCCIS data).

Parameters
----------

--debug: bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
--output: string (filepath)
    Filepath of where to save the output csv file. This is a required parameter. 
--att: string (filepath)
    Filepath of the attendance csv file. This is a required parameter.
--ks: string (filepath)
    Filepath of the KS4 csv file. This is a required parameter.
--census: string (filepath)
    Filepath of the census csv file. This is a required parameter.
--ccis: string (filepath)
    Filepath of the CCIS csv file. This is a required parameter.
--target: 
    Filepath of where to save the output csv file. This is a required parameter. 
--pre_age15 (bool)
    Whether to remove datapoints from >=15 years of age (year 11 data as we will not have this when predicting prior to year 11). Default is true.

Returns
-------
csv file
   multi upn merged dataset saved at output filepath as a csv file.  


"""


from dataclasses import asdict
import pandas as pd
import os
import argparse
from datetime import datetime
from enum import Enum

# DVC Params
from src.constants import (
    Targets,
    CCISDataColumns,
    CensusDataColumns,
    AttendanceDataColumns,
    SchoolInfoColumns,
    KSDataColumns,
    NA_VALS,
    UNKNOWN_CODES,
    UPN
)

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import merge_utils as mu
from src import ccis_utils as nu

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--output', required=True,
                    help='where to output the produced csv file')
parser.add_argument('--att', required=True,
                    help='where to find input premerge attendance csv')
parser.add_argument('--ks', required=True,
                    help='where to find input premerge key stage csv')
parser.add_argument('--census', required=True,
                    help='where to find input premerge census csv')
parser.add_argument('--ccis', required=True,
                    help='where to find input premerge ccis csv')
parser.add_argument('--target', required=True, choices=list(asdict(Targets).values()),
                    help='which target variable to add to csv')
parser.add_argument('--pre_age15', action='store_true',
                    help='whether or not to throw out all students aged >= 15')

def school_year_of_age(df, age: int):
    '''
    Returns school year based on age and birth month of student
    
    Parameters
    ----------
    df: pd.DataFrame
    age: int
    
    Returns
    ----------
    school_year: pd.Series
        School year based on age and birth month of student
    '''
    school_year = pd.Series([pd.NA] * len(df))
    school_year.name = CCISDataColumns.year
    # Student's age is their age by the end of Aug 31. Say the student age is 10.
    # So if their birth_month >= 9, then this is the year they turn 11, and it is (10+1) + birth_year.
    # If their birth_month < 9, then this is the year they turn 10, and the year is 10 + birth_year
    sept_mask = df[CCISDataColumns.birth_month] >= 9
    school_year[sept_mask] = df.loc[sept_mask, CCISDataColumns.birth_year] + (age + 1)
    school_year[~sept_mask] = df.loc[~sept_mask, CCISDataColumns.birth_year] + age
    assert school_year.isna().sum() == 0
    
    return school_year

def merge_multi_upn(neet_single_upn_df, census_df, ks_df, att_df, na_vals=NA_VALS, logger=l.PrintLogger):
    '''
    Merges 4 input dataframes (neet, census, key stage, attendance) and outputs merged dataframe with multiple years of data per upn.
    
    Parameters
    ----------
    neet_single_upn_df: pd.DataFrame 
    census_df: pd.DataFrame
    ks_df: pd.DataFrame
    att_df: pd.DataFrame
    na_vals: iterable (e.g. list)
    logger:
    
    Returns
    ----------
    pd.Dataframe
        merged multi upn dataframe 
    '''
    
    # Collect labels
    logger.info('Collecting labels from neet data')
    
    # Inner Merge with census data
    logger.info('Merging labels with census data')
    census_labeled_df = mu.merge_priority_data(
        df1=neet_single_upn_df, 
        df2=census_df,
        on=CCISDataColumns.upn, 
        how='inner',
        unknown_vals = UNKNOWN_CODES,
        na_vals=NA_VALS,
    )
    census_labeled_df[CensusDataColumns.has_census_data] = 1
    census_df = None
    
    # Inner merge with attendance data
    logger.info('Merging labels with attendance data')
    att_labeled_df = mu.merge_priority_data(
        df1=neet_single_upn_df, 
        df2=att_df, 
        on=CCISDataColumns.upn, 
        how='inner',
        unknown_vals = UNKNOWN_CODES,
        na_vals=NA_VALS,
    )
    att_labeled_df[AttendanceDataColumns.has_attendance_data] = 1
    
#     breakpoint()
    
    # Inner merge with ks data
    logger.info('Merging labels with ks data')
    ks_labeled_df = mu.merge_priority_data(
        df1=neet_single_upn_df,
        df2=ks_df,
        on=CCISDataColumns.upn, 
        how='inner',
        unknown_vals = UNKNOWN_CODES,
        na_vals=NA_VALS,
    )
    ks_labeled_df[KSDataColumns.has_ks2_data] = 1
    
    # Outer merges to bring data together
    logger.info('Merging upn-year datasets w/ priority')
    merged_df = mu.merge_priority_data(
        df1=census_labeled_df, 
        df2=att_labeled_df, 
        on=[CCISDataColumns.upn, CCISDataColumns.year], 
        how='outer',
        unknown_vals=UNKNOWN_CODES,
        na_vals=NA_VALS,
    )
#     breakpoint()
    census_labeled_df = None
    att_labeled_df = None
    
    # Finally merge in key stage data
    logger.info('Merging ks data to upns')
    merged_df = mu.merge_priority_dfs(
        [merged_df, ks_labeled_df], 
        on=CCISDataColumns.upn, 
        how='outer',
        unknown_vals=UNKNOWN_CODES,
        na_vals=NA_VALS,
    )
#     breakpoint()
    
    merged_df.loc[:, [
        CensusDataColumns.has_census_data, 
        AttendanceDataColumns.has_attendance_data,
        KSDataColumns.has_ks2_data
    ]] = merged_df[[
        CensusDataColumns.has_census_data, 
        AttendanceDataColumns.has_attendance_data,
        KSDataColumns.has_ks2_data
    ]].fillna(0)
    
#     breakpoint()
    
    # Any remaining nan years are because those students weren't in the census data or the attendance data.
    # Since we only have key stage 2 data, we'll fill in their year as 10 years after birth 
    # (they'll start the school year as 10yrs old then turn 11). So if their birth month >= 9, it's 11 yrs + birth year,
    # otherwise it's 10 yrs + birth_year
    age_fill_in = 10
    logger.info(f'Filling in age {age_fill_in} for students with only ks2 data')
    merged_df[CCISDataColumns.year].fillna(school_year_of_age(merged_df, age=age_fill_in), inplace=True)
    
    # Drop unnecessary columns TODO: do these inplace
    logger.info('Dropping unnecessary columns')
    merged_df = d.drop_empty_cols(merged_df, na_vals=na_vals, logger=logger)
    merged_df = d.drop_single_valued_cols(merged_df, logger=logger)
    merged_df = d.drop_duplicate_rows(merged_df, logger=logger)
    merged_df = merged_df.reset_index(drop=True)
    
    merged_df = merged_df.convert_dtypes()
    
#     breakpoint()
    
    return merged_df

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    # breakpoint()
    if args.target in [Targets.neet_ever, Targets.neet_unknown_ever]:
        logger.info(f'Building multi-upn dataset with target variable {args.target}')
    elif args.target in [Targets.unknown_ever]:
        logger.error(f'Target {args.target} is not implemented yet.')
        raise NotImplementedError()
    else:
        logger.error(f'Unknown target variable {args.target}')
        raise ValueError()
    
    att_df = d.load_csv(
        args.att, 
        drop_empty=True, 
        drop_single_valued=True,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        na_vals=NA_VALS,
        upn_col=UPN,
        logger=logger
    )
    ks_df = d.load_csv(
        args.ks, 
        drop_empty=True, 
        drop_single_valued=True,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        na_vals=NA_VALS,
        upn_col=UPN,
        logger=logger
    )
    neet_df = d.load_csv(
        args.ccis, 
        drop_empty=True, 
        drop_single_valued=True,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        na_vals=NA_VALS,
        upn_col=UPN,
        logger=logger
    )
    census_df = d.load_csv(
        args.census, 
        drop_empty=True, 
        drop_single_valued=True,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        na_vals=NA_VALS,
        upn_col=UPN,
        logger=logger
    )
    
#     breakpoint()
    
    merged_df = merge_multi_upn(
        neet_single_upn_df=neet_df, 
        census_df=census_df, 
        ks_df=ks_df, 
        att_df=att_df,
        na_vals = NA_VALS,
        logger=logger
    )
    
    # breakpoint()
#     
    # Set to None so python garbage collector will come clean them up
    neet_df = None
    census_df = None
    att_df = None
    ks_df = None
    
    logger.info(f'Initial merged row count {len(merged_df)}')
    logger.info(f'Initial column count {len(merged_df.columns)}')
    
    # Convert to int since it went to float for some reason
    to_int = [
        CCISDataColumns.neet_ever, 
        CCISDataColumns.unknown_ever,
        CCISDataColumns.year,
        CCISDataColumns.birth_year,
        CCISDataColumns.birth_month,
    ]
    logger.info(f'Converting columns {to_int} back to int format')
    merged_df.loc[:, to_int] = merged_df.loc[:, to_int].astype(int)
    
#     
    
    logger.info(f'Computing the age values and dropping birth year column')
    merged_df[CCISDataColumns.age] = nu.compute_age(merged_df)
    merged_df.drop(CCISDataColumns.birth_year, axis=1, inplace=True)  
    
    # breakpoint()
    
    if args.pre_age15:
        logger.info(f'Dropping samples with age >= 15')
        merged_df = merged_df[merged_df[CCISDataColumns.age] < 15]
    
    # breakpoint()
    # Do target specific processing
    if args.target == Targets.neet_ever:
        
        logger.info('Dropping students who have unknown status but were never neet')
        merged_df = merged_df[merged_df[CCISDataColumns.neet_ever].astype(bool) | (~merged_df[CCISDataColumns.unknown_ever].astype(bool))]
        
    if args.target == Targets.neet_unknown_ever:
        logger.info('Doing NEET & unknown target specific processing')
        logger.info('Creating column for neet/unknown ever that is 1 if the student was ever neet or unknown')
        merged_df[CCISDataColumns.neet_unknown_ever] = (merged_df[CCISDataColumns.neet_ever].astype(bool) | merged_df[CCISDataColumns.unknown_ever].astype(bool)).astype(int)
    
    # breakpoint()
        
    logger.info(f'Final merged row count {len(merged_df)}')
    census_count = merged_df[CensusDataColumns.has_census_data].sum()
    ks_count = merged_df[KSDataColumns.has_ks2_data].sum()
    att_count = merged_df[AttendanceDataColumns.has_attendance_data].sum()
    logger.info(f'Final census count in merged {census_count}/{len(merged_df)} ({census_count / len(merged_df)})')
    logger.info(f'Final KS count in merged {ks_count}/{len(merged_df)} ({ks_count / len(merged_df)})')
    logger.info(f'Final att count in merged {att_count}/{len(merged_df)} ({att_count / len(merged_df)})')
    logger.info(f'Final column count {len(merged_df.columns)}')
    
#     breakpoint()
    
    csv_fp = args.output
    if args.debug:
         csv_fp = f.tmp_path(csv_fp)
    
    logger.info(f'Saving multi upn merged data to {csv_fp}')
    merged_df.to_csv(csv_fp, index=False)

    