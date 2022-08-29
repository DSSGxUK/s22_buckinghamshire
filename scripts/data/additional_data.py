"""
Prepare the dataset of additional data to merge back in with the final model predictions output csv 
We do this because some columns are removed for modelling that we need for the power bi dashboard 

These inputs will be filepaths of the datasets used for predicting.

Parameters
-----------
--debug : bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
--single : bool
    Whether the single upn dataset is used
--census_input : string
    The filepath where the annotated census data is saved
--att_input : string
    The filepath where the annotated attendance data is saved
--output : string
    The filepath where the output csv will be saved.

Returns
----------
csv file
    csv file containing the additional information to merge back with the final model predictions dataset
"""

import argparse

# DVC Params
from src.constants import (
    CensusDataColumns,
    SchoolInfoColumns,
    NA_VALS,
    UPN,
    AttendanceDataColumns
)

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import attendance_utils as au

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--single', action='store_true',
                    help='whether to use single upn dataset')
parser.add_argument('--census_input', required=True,
                    help='where to find input annotated census csv')
parser.add_argument('--att_input', required=True,
                    help='where to find input annotated attendance csv')
parser.add_argument('--output', required=True,
                    help='where to output the produced csv file')

def create_attendance_exact_df(df, logger=l.PrintLogger):
    """
    Groups attendance data by UPN and year to sum total number of absence days and termly sessions possible in each year for each student. 
    
    Parameters
    -----------
    df: pd.DataFrame
        Dataframe containing attendance data with separate term data for each student
    logger: logging.Logger
    
    Returns
    -----------
    pd.DataFrame
        dataframe with attendance data aggregated so only one attendance row per year for each student   
    
    """
    gby_columns=[AttendanceDataColumns.upn, AttendanceDataColumns.year]
    
    absence_columns = au.get_absence_reason_columns(df)
    nonabsence_columns = au.get_nonabsence_reason_columns(df)
    
    logger.info('Drop all columns that are not useful for attendance numbers')
    drop_columns = [c for c in df.columns if not au.keep_column_criteria(
        c, 
        gby_columns
    )]
    logger.info(f'Dropping {drop_columns}')
    df = df.drop(drop_columns, axis=1)
    
    logger.info('Do a groupby and sum')
    # The only columns left that are not na should all be numerical columns
    numerical_cols = [c for c in df.columns if c not in gby_columns]
    logger.info(f'Summing over columns {numerical_cols}')
    df.loc[:, numerical_cols] = d.to_int(df[numerical_cols])
    df.drop(df[df['total_absences']>df['termly_sessions_possible']].index, inplace=True) #drop rows in which total absences is greater than termly sessions possible
    df = df.groupby(by=gby_columns).sum().reset_index()
    
    return df

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    if args.single :
        df = d.load_csv(  # Modeling dataset so we drop unnecessary columns and entries
            args.census_input, 
            drop_empty=False, 
            drop_single_valued=False,
            drop_duplicates=True,
            read_as_str=False,
            drop_missing_upns=True,
            upn_col=UPN,
            na_vals=NA_VALS,
            logger=logger
        )

    #     breakpoint()

        # columns we want to keep for the additional dataset
        extra_columns = [UPN,
            CensusDataColumns.forename,
            CensusDataColumns.preferred_surname,
            CensusDataColumns.postcode,
            CensusDataColumns.entry_date,
            SchoolInfoColumns.establishment_area,
            SchoolInfoColumns.establishment_electoral_wards,
            SchoolInfoColumns.establishment_name,
            SchoolInfoColumns.establishment_number,
            SchoolInfoColumns.establishment_postcode,
            SchoolInfoColumns.la_establishment_number,
            CensusDataColumns.census_period_end,
            CensusDataColumns.nc_year_actual,
            CensusDataColumns.age 
        ]
        logger.info(f'Saving columns {extra_columns} from {args.census_input} to an additional dataset')

        extra_cols_df = df[extra_columns]
        extra_cols_df = extra_cols_df.sort_values(by=[CensusDataColumns.census_period_end,CensusDataColumns.nc_year_actual])
        
        logger.info(f'Aggregating {extra_columns} for single upn dataset')
        
        ##students shouldn't be  > year 10 in this dataset / older than 14
        census_agg = extra_cols_df.groupby(UPN).agg({'postcode':'last','forename':'last','preferred_surname':'last','establishment_name':'last','establishment_number':'last','establishment_postcode':'last','la_establishment_number':'last','nc_year_actual':'max','age':'max'}).reset_index()
        
        att_df = d.load_csv(  # Modeling dataset so we drop unnecessary columns and entries
            args.att_input, 
            drop_empty=True, 
            drop_single_valued=True,
            drop_duplicates=True,
            read_as_str=False,
            drop_missing_upns=True,
            upn_col=UPN,
            na_vals=NA_VALS,
            logger=logger
        )    

        logger.info(f'Calculating exact attendance data from {args.att_input}')
        att_exact = create_attendance_exact_df(att_df, logger=l.PrintLogger)
        logger.info(f'Aggregating number of days absent for each student {args.att_input}')

        att_agg = att_exact.groupby(UPN).agg({'excluded_authorised':'sum'}).reset_index()
        #att_exact[]
        logger.info(f'Merging census and attendance extra columns dataset')
        
        merged = census_agg.merge(att_agg, how='outer',on=UPN)
        merged = merged.rename(columns={'excluded_authorised':'excluded_authorised_exact'})                
        #breakpoint()

        csv_fp = args.output
        if args.debug:
             csv_fp = f.tmp_path(csv_fp)

        logger.info(f'Saving data to {csv_fp}')
        merged.to_csv(csv_fp, index=False)
    
        #breakpoint()
    
    else :
        logger.error("We don't have code yet to test the multi-upn dataset")
        raise NotImplementedError()
