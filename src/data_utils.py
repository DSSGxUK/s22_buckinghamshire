"""
This utils file contains helpers for processing the data files
"""

from typing import List, Dict
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from re import compile as re_compile

from .constants import (
    NA_VALS,
    UPN,
    CATEGORICAL_SEP
)

from . import log_utils as l
from .error_utils import error_with_logging

def read_excel_file(
    xl_fp: str,
    header_row=0, 
    sheet_name=None,
    logger=l.PrintLogger
):
    logger.info(f"Reading {xl_fp}")
    with pd.ExcelFile(xl_fp) as xl:
        read_sheet_name = sheet_name
        if read_sheet_name is None:
            # By default assume that there is only one sheet per excel file
            assert len(xl.sheet_names) == 1
            read_sheet_name = 0
            logger.debug(f"Only 1 sheet")
        elif isinstance(read_sheet_name, int):
            assert 0 <= read_sheet_name < len(xl.sheet_names)
            logger.info(f"Reading sheet index {read_sheet_name}")
        else:
            assert read_sheet_name in xl.sheet_names
            logger.info(f"Reading sheet {read_sheet_name}")
    
        # Sometimes bad excel formulas will be read as na
        df = pd.read_excel(xl, sheet_name=read_sheet_name, keep_default_na=False, na_values=[], dtype=str, header=header_row) \
                    .fillna('')
    return df


def read_excel_files(
    xl_fps: Dict[str, str], 
    header_row=0, 
    sheet_name=None,
    logger=l.PrintLogger
):
    dfs = {}
    for yr, fp in xl_fps.items():
        logger.info(f"Reading key {yr}")
        dfs[yr] = read_excel_file(fp, header_row=header_row, sheet_name=sheet_name, logger=logger)
    return dfs

def read_excel_date(excel_date):
    excel_date = int(excel_date)
    return (datetime.strptime('1899-12-30', "%Y-%m-%d") + timedelta(excel_date)).strftime("%Y-%m-%d %H:%M:%S")

def normalize_date(datetime_str: str, logger=l.PrintLogger) -> str:
    output_fmt = "%Y-%m-%d %H:%M:%S"
    formats = ["%Y-%m-%d", output_fmt]
    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt).strftime(output_fmt)
        except:
            continue
    
    # None of those worked, so we try to read as excel date
    try:
        return read_excel_date(datetime_str)
    except:
        pass

    error_with_logging(
        f"Could not parse date string {datetime_str}",
        logger,
        ValueError
    )



def isna(x, na_vals=NA_VALS):
    return x.isin(na_vals) | x.isna()

def isna_scalar(x, na_vals=NA_VALS):
    return pd.isna(x) or (x in na_vals)

def dropna(x, na_vals=NA_VALS):
    return x[~isna(x, na_vals=na_vals)]

def to_int(x, fillna=0, na_vals=NA_VALS):
    x = x.copy()
    x[isna(x, na_vals=na_vals)] = fillna
    return x.astype(int)

def safe_drop_columns(df: pd.DataFrame, columns: List, inplace=False) -> pd.DataFrame:
    """Will drop columns any of the input columns from df. Will continue if a column does not exist"""
    to_drop = list(set(columns) & set(df.columns))
    if inplace:
        df.drop(to_drop, axis=1, inplace=True)
        return df
    else:
        return df.drop(to_drop, axis=1, inplace=False)

def save_xl_to_csv(
    xl_fp: str,
    csv_fp: str,
    header_row=0,
    sheet_name=None,
    logger=l.PrintLogger,
):
    df = read_excel_file(xl_fp, header_row=header_row, sheet_name=sheet_name, logger=logger)
    logger.info(f"Writing {xl_fp} to {csv_fp}")
    df.to_csv(csv_fp, index=False)

def save_xls_to_csv(
    xl_fps: Dict[str, str], 
    csv_fps: Dict[str, str], 
    preloaded_dfs = None,
    header_row=0,
    sheet_name=None,
    logger=l.PrintLogger
): 
    if preloaded_dfs is None:
        dfs = read_excel_files(xl_fps, header_row=header_row, sheet_name=sheet_name, logger=logger)
    else:
        dfs = preloaded_dfs
    for k, csv_path in csv_fps.items():
        xl_path = xl_fps[k]
        df = dfs[k]
        logger.info(f"Writing {xl_path} to {csv_path} for key {k}")
        df.to_csv(csv_path, index=False)

def get_unique_nonna(df, col, na_vals=NA_VALS):
    the_col = df[col]
    # TODO Replace this with isna
    return set((the_col[~the_col.isin(na_vals)].dropna()).unique())

def load_csvs(
    fps: Dict[str, str], 
    drop_empty=False, 
    drop_single_valued=False, 
    drop_missing_upns=False, 
    drop_duplicates=False, 
    read_as_str=False,
    upn_col=UPN,
    na_vals=NA_VALS,
    logger=l.PrintLogger
):
    return_dict = {}
    for k, fp in fps.items():
        logger.info(f"Loading CSV at {fp} for {k}")
        return_dict[k] = load_csv(
            fp, 
            drop_empty=drop_empty,
            drop_single_valued=drop_single_valued,
            drop_missing_upns=drop_missing_upns, 
            drop_duplicates=drop_duplicates,
            read_as_str=read_as_str,
            upn_col=upn_col,
            na_vals=na_vals,
            logger=logger,
        )
    return return_dict
            

def load_csv(
    fp, 
    drop_empty=False, 
    drop_single_valued=False, 
    drop_missing_upns=False, 
    drop_duplicates=False, 
    read_as_str=False,
    use_na=False,
    upn_col=UPN,
    na_vals=NA_VALS,
    convert_dtypes=False,  # This will use the experimental pd.NA. Code from pre_age15_multi_upn_categorical will try to use this. If read_as_str=True, then all columns will be turned into strings. Otherwise, it will infer types
    downcast=False,
    logger=l.PrintLogger
):
    logger.info(f"Reading {fp}")
    if read_as_str:
        logger.info(f'Reading all data as str')
        df = pd.read_csv(fp, keep_default_na=False, dtype=str)
    else:
        df = pd.read_csv(fp, keep_default_na=False, na_values=na_vals)
    if use_na:
        logger.info(f'Replacing {na_vals} with np.nan')
        df = df.replace({k: np.nan for k in na_vals})
    if convert_dtypes:
        logger.info(f'Converting pandas dtypes and using pd.NA instead of np.nan')
        df = df.convert_dtypes()\
            .fillna(pd.NA)\
            .convert_dtypes()
    if downcast:
        logger.info(f'Downcasting dataframe to save space')
        df = downcast_df(df, inplace=True)
    if drop_empty:
        df = drop_empty_cols(df, na_vals=na_vals, logger=logger)
    if drop_single_valued:
        df = drop_single_valued_cols(df, logger=logger)
    if drop_duplicates:
        df = drop_duplicate_rows(df, logger=logger)
    if drop_missing_upns:
        df = drop_empty_upns(df, upn_col, logger=logger)
    df = df.reset_index(drop=True)
    return df

# Deprecated
# def load_csv_of_type(
#     fp, 
#     df_type,
#     drop_empty=True, 
#     drop_single_valued=True, 
#     drop_missing_upns=False, 
#     drop_duplicates=True, 
#     upn_col=UPN,
# ):
#     df = load_csv(
#         fp,
#         drop_empty=drop_empty, 
#         drop_single_valued=drop_single_valued, 
#         drop_missing_upns=drop_missing_upns, 
#         drop_duplicates=drop_duplicates, 
#         upn_col=UPN,
#     )
#     if df_type == constants.CENSUS:
#         print('Processing special logic for Census data')
#         df['SENneed1'] = df['SENneed1'].fillna(df['SENNeed1'])
#         df['SENneed2'] = df['SENneed2'].fillna(df['SENNeed2'])
#         df = df.drop(['SENNeed1', 'SENNeed2'], axis=1)
#     else:
#         print(f'No special logic to process for {df_type} data')
    
#     return df
        

# Deprecated
# def load_all_merged_data(
#     drop_empty=True, 
#     drop_single_valued=True, 
#     drop_missing_upns=False, 
#     drop_duplicates=True,
#     should_rename_codes=True
# ):
#     df_dict = {k: load_csv_of_type(
#         fp, 
#         df_type=k,
#         drop_empty=drop_empty, 
#         drop_single_valued=drop_single_valued, 
#         drop_missing_upns=drop_missing_upns,
#         drop_duplicates=drop_duplicates
#     ) for k, fp in constants.MERGED_FPS.items()}
    
#     if should_rename_codes:
#         codes_dict = load_codes()
#         print('Renaming codes')
#         df_dict = rename_codes(df_dict, codes_dict)
    
#     return df_dict
    

def drop_duplicate_rows(df, logger=l.PrintLogger):
    new_df = df.drop_duplicates()
    n_dropped_rows = len(df) - len(new_df)
    logger.info(f'Dropped {n_dropped_rows} that were duplicates of other rows.')
    return new_df
            
def drop_empty_cols(df, na_vals=NA_VALS, logger=l.PrintLogger):
    new_df = df.loc[:, ~(df.isna() | df.isin(na_vals)).all(axis=0)]
    dropped_cols = set(df.columns) - set(new_df.columns)
    logger.info(f'Dropped columns {dropped_cols} because they were empty')
    return new_df

def drop_single_valued_cols(df, logger=l.PrintLogger):
    nunique = df.nunique(dropna=False)
    cols_to_drop = nunique[nunique == 1].index
    new_df = df.drop(cols_to_drop, axis=1)
    logger.info(f'Dropped columns {cols_to_drop} because they had only one value')
    return new_df

def drop_empty_upns(df, upn_col, na_vals=NA_VALS, logger=l.PrintLogger):
    new_df = df[~isna(df[upn_col], na_vals=na_vals)]
    removed_rows = len(df) - len(new_df)
    logger.info(f'Removed {removed_rows} rows that were missing UPNs')
    return new_df

def downcast_df(df, inplace=False):
    if not inplace:
        df = df.copy()
    fcols = df.select_dtypes('float').columns #returns columns based on dtype
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    
    return df

def all_equal(df):
    # Vacuously all are equal if there are no elements
    return df.nunique(dropna=False) <= 1

# Deprecated
# def differing_duplicate_upns_by_year(df, dataset):
#     assert dataset in constants.YEAR_COLS
#     df_groups = df.groupby(by=[UPN, constants.YEAR_COLS[dataset]])
#     duplicate_differences_df = df_groups.aggregate(all_equal)
#     duplicate_differences_df['differing_columns'] = duplicate_differences_df \
#         .apply(lambda s: set(s.index[~s.astype(bool)]), axis=1)
#     return duplicate_differences_df[['differing_columns']].reset_index()

def count_duplicates(df):
    return len(df)-len(df.drop_duplicates())

def create_code_dict(codes_df, key_col, val_col, prefix=''):
    code_dict = {}
    for _, (k, v) in codes_df[[key_col, val_col]].iterrows():
        if isna_scalar(k):
            continue
        code_dict[k] = prefix + v
    return code_dict

def create_code_dicts(codes_df):
    activity_code_dict = create_code_dict(codes_df, 'activity_code', 'activity_name', prefix='Activity: ')
    characteristic_code_dict = create_code_dict(codes_df, 'characteristic_code', 'characteristic_name', prefix='Characteristic: ')
    intended_dest_code_dict = create_code_dict(codes_df, 'cohort_status', 'cohort_name', prefix='Cohort Status: ')
    cohort_status_code_dict = create_code_dict(codes_df, 'intended_yr11_destination', 'intended_yr11_destination_name', prefix='Intended Yr11 Destination: ')
    enrol_status_code_dict = create_code_dict(codes_df, 'enrol_status', 'enrol_status_name', prefix='Enrol Status: ')
    sen_need_code_dict = create_code_dict(codes_df, 'sen_need', 'sen_need_name', prefix='SEN Need: ')
    ethnicity_code_dict = create_code_dict(codes_df, 'ethnicity', 'ethnicity_name', prefix='Ethnicity: ')
    language_code_dict = create_code_dict(codes_df, 'language', 'language_name', prefix='Language: ')
    level_of_need_code_dict = create_code_dict(codes_df, 'level_of_need_code', 'level_of_need_name', prefix='Level of Need: ')
    return {
        'activity_code': activity_code_dict,
        'characteristic_code': characteristic_code_dict,
        'cohort_status': intended_dest_code_dict,
        'intended_yr11_destination': cohort_status_code_dict,
        'enrol_status': enrol_status_code_dict,
        'sen_need': sen_need_code_dict,
        'ethnicity': ethnicity_code_dict,
        'language': language_code_dict,
        'level_of_need_code': level_of_need_code_dict,
    }

# Deprecated
# def load_codes(codes_fp=constants.NEET_CODES_FP):
#     print(f'Loading codes dict from {codes_fp}')
#     codes_df = pd.read_csv(codes_fp)
#     return create_code_dicts(codes_df)

# def rename_codes(df_dict, codes_dict):
#     new_df_dict = {}
#     for df_type, df in df_dict.items():
#         if df_type == constants.NEET:
#             print(f'Renaming codes in {df_type} data')
#             new_df_dict[df_type] = rename_codes_neet(df, codes_dict)
#         elif df_type == constants.CENSUS:
#             print(f'Renaming codes in {df_type} data')
#             new_df_dict[df_type] = rename_codes_census(df, codes_dict)
#         else:
#             print(f'No codes to rename in {df_type} data')
#             new_df_dict[df_type] = df
#     return new_df_dict

        
def _rename_individual_codes(df, codes_dict, rename_cols):
    df = df.copy()
    for code_dict_key, source_col, new_col in rename_cols:
        code_dict_type = type(next(iter(codes_dict[code_dict_key].keys()))) # Just use the first one.
        df[new_col] = df[source_col].apply(lambda x: x if isna_scalar(x) else codes_dict[code_dict_key][code_dict_type(x)])
    return df

def rename_codes_neet(df, codes_dict):
    rename_cols = [
        ('activities', 'ActivityCode', 'ActivityName'),
        ('characteristics', 'CharacteristicCode', 'CharacteristicName'),
        ('intended_yr11_destinations', 'IntendedDestinationYr11', 'IntendedYr11DestinationName'),
        ('cohorts', 'CohortStatus', 'CohortName'),
    ]
    return _rename_individual_codes(df, codes_dict, rename_cols)

def rename_codes_census(df, codes_dict):
    df = df.copy()
    rename_cols = [
        ('enrolment', 'EnrolStatus', 'EnrolStatusName'),
        ('sen_need', 'SENneed1', 'SENneedName1'),
        ('sen_need', 'SENneed2', 'SENneedName2')
    ]
    return _rename_individual_codes(df, codes_dict, rename_cols)

def peek_iterable(iterable):
    return next(iter(iterable))

def groupby_to_dict(gby):
    return {k: v for k,v in gby}

def map_groupings(grouping_dict, f, default=None, progress=False):
    prog = tqdm if progress else lambda x: x
    new_dict = {k : f(v) for k, v in prog(grouping_dict.items())}
    if default is not None:
        new_dict = defaultdict(default, new_dict)
    return new_dict

def add_column(df, col_name, data):
    df = df.copy()
    df[col_name] = data
    return df

def savedate_to_datetime(date):
    return datetime.strptime(date, '%b%y')

def filter_columns(df, cols):
    return df[[c for c in cols if c in df.columns]]

NUMBER_REGEX = re_compile("^\d+?\.\d+?$")
def is_number_scalar(x: str):
    if isna_scalar(x):
        return False
    if NUMBER_REGEX.match(x) is None:
        return x.isdigit()
    return True

def is_number(x: pd.Series):
    return x.astype(str).apply(is_number_scalar)

def divide_with_na(x, y, fillna=0, fill_prev_na=False):
    res = x / y
    if fill_prev_na:
        res = res.fillna(fillna)
    if not fill_prev_na:
        res.loc[~x.isna()] = res.loc[~x.isna()].fillna(fillna)
    return res

def get_dummies_with_logging(df, columns, prefix_sep=CATEGORICAL_SEP, logger=l.PrintLogger):
    """Provides some logging through when applying `pd.get_dummies`.

    We use this method as a shorthand because we often want to see
    what columns got created during the application of `pd.get_dummies`.
    It can be useful for debugging.

    For more info on pd.get_dummies, please see the documentation:
    https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to apply `pd.get_dummies` to.
    columns : iterable
        The columns to pass to `pd.get_dummies`. Usually a list.
    prefix_sep: str
        The separator to pass to `pd.get_dummies`.
    logger : logging.Logger, optional
        The logger with which to log the process. Defaults to
        a `log_utils.PrintLogger`.

    Returns
    -------
    pd.DataFrame
        The result of calling `pd.get_dummies(df, columns=columns, prefix_sep=prefix_sep)`.
        See pandas documentation (linked above) for more details.

    TODO: Unit test    
    """
    logger.info(f'Creating categorical columns for columns {columns}')
    prev_columns = set(df.columns)
    df = pd.get_dummies(df, columns=columns, prefix_sep=prefix_sep)
    created_columns = list(set(df.columns) - prev_columns)
    logger.info(f'Created categorical columns {created_columns}')

    return df





    