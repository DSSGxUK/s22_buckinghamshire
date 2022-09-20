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
import argparse

# DVC Params
from src.constants import (
    Targets,
    CCISDataColumns,
    CharacteristicsDataColumns,
    CensusDataColumns,
    AttendanceDataColumns,
    OutputDatasetTypes,
    KSDataColumns,
    KS2Columns,
    NA_VALS,
    UNKNOWN_CODES,
    UPN,
    YEAR,
    AGE,
    CATEGORICAL_SEP,
    non_prediction_columns,
    PupilDeprivationColumns,
)

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import merge_utils as mu
from src import ccis_utils as nu
from src import py_utils as py

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--output", type=lambda x: x.strip("'"), required=True, help="where to output the produced csv file"
)
parser.add_argument(
    "--att", type=lambda x: x.strip("'"), required=True, help="where to find input premerge attendance csv"
)
parser.add_argument(
    "--ks2",
    type=lambda x: x.strip("'"), required=False,  # This can be None, in which case we won't use
    help="where to find input premerge key stage 2 csv",
)
parser.add_argument(
    "--census", type=lambda x: x.strip("'"), required=True, help="where to find input premerge census csv"
)
parser.add_argument(
    "--ccis", type=lambda x: x.strip("'"), required=True, help="where to find input premerge ccis csv"
)
parser.add_argument(
    "--output_dataset_type",
    type=lambda x: x.strip("'"), required=True,
    choices=asdict(OutputDatasetTypes).values(),
    help="What kind of output merged dataset to build",
)
parser.add_argument("--only_att_census_merge", action="store_true", help="whether to only use attendance and census features")
# Optional in case we don't have it
parser.add_argument(
    "--chars", type=lambda x: x.strip("'"), required=False, help="where to find the input characteristics data"
)
parser.add_argument(
    "--target",
    type=lambda x: x.strip("'"), required=False,
    choices=list(asdict(Targets).values()),
    help="which target variable to add to csv",
)


def school_year_of_age(df):
    """
    Returns school year based on age and birth month of student

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe with CCIS birth_year and birth_month columns and a birth year column

    Returns
    ----------
    school_year: pd.Series:
        School year based on age and birth month of student
    """
    school_year = d.empty_series(len(df), name=YEAR)
    # Student's age is their age by the end of Aug 31. Say the student age is 10.
    # So if their birth_month >= 9, then this is the year they turn 11, and it is (10+1) + birth_year.
    # If their birth_month < 9, then this is the year they turn 10, and the year is 10 + birth_year
    sept_mask = df[CCISDataColumns.birth_month] >= 9
    school_year[sept_mask] = df.loc[sept_mask, CCISDataColumns.birth_year] + (
        df[AGE] + 1
    )
    school_year[~sept_mask] = df.loc[~sept_mask, CCISDataColumns.birth_year] + df[AGE]

    return school_year


if __name__ == "__main__":
    args = parser.parse_args()
    # Validate the arguments
    # This would have been better with subparsers, but they weren't working when used with common required arguments as well
    if args.output_dataset_type == OutputDatasetTypes.modeling:
        assert args.target is not None
        assert args.chars is None
    if args.output_dataset_type == OutputDatasetTypes.prediction:
        assert args.target is None
    if args.output_dataset_type == OutputDatasetTypes.unknowns:
        assert args.target is None
        assert args.chars is None

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    if args.output_dataset_type == OutputDatasetTypes.modeling:
        if args.target in [Targets.neet_ever, Targets.neet_unknown_ever]:
            logger.info(
                f"Building multi-upn dataset with target variable {args.target}"
            )
        elif args.target in [Targets.unknown_ever]:
            logger.error(f"Target {args.target} is not implemented yet.")
            raise NotImplementedError()

    # Should not be any duplicated rows or rows with missing upns
    att_df = d.load_csv(
        args.att,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=False,
        read_as_str=False,
        drop_missing_upns=False,
        na_vals=NA_VALS,
        upn_col=UPN,
        use_na=True,
        convert_dtypes=True,
        downcast=True,
        logger=logger,
    )
    if args.ks2 is not None:
        ks2_df = d.load_csv(
            args.ks2,
            drop_empty=False,
            drop_single_valued=False,
            drop_duplicates=False,
            read_as_str=False,
            drop_missing_upns=False,
            na_vals=NA_VALS,
            upn_col=UPN,
            use_na=True,
            convert_dtypes=True,
            downcast=True,
            logger=logger,
        )
    else:
        ks2_df = pd.DataFrame(
            columns=list(asdict(KS2Columns).values() | asdict(PupilDeprivationColumns).values())
        )

    neet_df = d.load_csv(
        args.ccis,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=False,
        read_as_str=False,
        drop_missing_upns=False,
        na_vals=NA_VALS,
        upn_col=UPN,
        use_na=True,
        convert_dtypes=True,
        downcast=True,
        logger=logger,
    )
    census_df = d.load_csv(
        args.census,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=False,
        read_as_str=False,
        drop_missing_upns=False,
        na_vals=NA_VALS,
        upn_col=UPN,
        use_na=True,
        convert_dtypes=True,
        downcast=True,
        logger=logger,
    )
    if (
        args.output_dataset_type == OutputDatasetTypes.prediction
        and args.chars is not None
    ):
        chars_df = d.load_csv(
            args.chars,
            drop_empty=False,
            drop_single_valued=False,
            drop_duplicates=False,
            read_as_str=False,
            drop_missing_upns=False,
            na_vals=NA_VALS,
            upn_col=UPN,
            use_na=True,
            convert_dtypes=True,
            downcast=True,
            logger=logger,
        )
    else:
        chars_df = pd.DataFrame(
            columns=d.expand_categorical_columns(
                asdict(CharacteristicsDataColumns).values(),
                neet_df.columns,
                prefix_sep=CATEGORICAL_SEP,
            )
        )

    ### UPN Filter
    neet_df[
        [
            CCISDataColumns.compulsory_school_always,
            CCISDataColumns.neet_ever,
            CCISDataColumns.unknown_ever,
            CCISDataColumns.unknown_currently,
        ]
    ] = (
        neet_df[
            [
                CCISDataColumns.compulsory_school_always,
                CCISDataColumns.neet_ever,
                CCISDataColumns.unknown_ever,
                CCISDataColumns.unknown_currently,
            ]
        ]
        .astype(pd.Int8Dtype())
        .astype(pd.BooleanDtype())
    )

    if args.output_dataset_type == OutputDatasetTypes.modeling:
        logger.info(
            f"Output dataset type is {OutputDatasetTypes.modeling} so only keeping students data after compulsory school and either a neet activity or no unknown activity"
        )
        if args.only_att_census_merge : #only keep essential columns in ks2 and ccis data
            neet_df = neet_df[[UPN,CCISDataColumns.neet_ever,CCISDataColumns.unknown_ever,CCISDataColumns.compulsory_school_always,CCISDataColumns.unknown_currently,CCISDataColumns.birth_month,CCISDataColumns.birth_year]]
            ks2_df = ks2_df[[UPN]]
            
        neet_df = neet_df[~neet_df[CCISDataColumns.compulsory_school_always]]
        if args.target == Targets.neet_ever:
            neet_df = neet_df[
                neet_df[CCISDataColumns.neet_ever]
                | ~(neet_df[CCISDataColumns.unknown_ever])
            ]
        elif args.target == Targets.neet_unknown_ever:
            neet_df[CCISDataColumns.neet_unknown_ever] = (
                neet_df[CCISDataColumns.neet_ever]
                | neet_df[CCISDataColumns.unknown_ever]
            )

        logger.info(f"Keeping {neet_df[UPN].nunique()} UPNs")
        # Now remove from other dfs
        census_df = census_df.merge(neet_df[UPN], on=UPN, how="inner")
        att_df = att_df.merge(neet_df[UPN], on=UPN, how="inner")
        ks2_df = ks2_df.merge(neet_df[UPN], on=UPN, how="inner")
        # Now remove all upns from neet
        other_upns = (
            set(census_df[UPN].unique())
            | set(att_df[UPN].unique())
            | set(ks2_df[UPN].unique())
        )
        neet_df = neet_df[neet_df[UPN].isin(other_upns)]

    elif args.output_dataset_type == OutputDatasetTypes.unknowns:
        logger.info(
            f"Output dataset type is {OutputDatasetTypes.unknowns} so only keeping students data are currently unknown"
        )
        neet_df = neet_df[neet_df[CCISDataColumns.unknown_currently]]
        logger.info(f"Keeping {neet_df[UPN].nunique()} UPNs")
        # Now remove from other dfs
        census_df = census_df.merge(neet_df[UPN], on=UPN, how="inner")
        att_df = att_df.merge(neet_df[UPN], on=UPN, how="inner")
        ks2_df = ks2_df.merge(neet_df[UPN], on=UPN, how="inner")

    elif args.output_dataset_type == OutputDatasetTypes.prediction:
        logger.info(
            f"Output dataset type is {OutputDatasetTypes.prediction} so only keeping students who are in the current census or attendance data and not in ccis data"
        )
        latest_year = max(att_df[YEAR].max(), census_df[YEAR].max())
        logger.info(f"Latest year in dataset is {latest_year}")
        census_df = census_df[census_df[YEAR] == latest_year]
        att_df = att_df[att_df[YEAR] == latest_year]
        chars_df = chars_df[chars_df[YEAR] == latest_year]

        logger.info("Constructing UPNs to keep.")
        # Don't include ks2 students in the upn set because KS2 could be from many years back.
        without_ccis_upns = pd.DataFrame(
            list((set(census_df[UPN]) | set(att_df[UPN]) | set(chars_df[UPN])) - set(neet_df[UPN])),
            columns=[UPN],
        )
        logger.info(f"Keeping {without_ccis_upns[UPN].nunique()} UPNs")
        # Now remove from other dfs
        census_df = census_df.merge(without_ccis_upns[UPN], on=UPN, how="inner")
        att_df = att_df.merge(without_ccis_upns[UPN], on=UPN, how="inner")
        ks2_df = ks2_df.merge(without_ccis_upns[UPN], on=UPN, how="inner")
        chars_df = chars_df.merge(without_ccis_upns[UPN], on=UPN, how="inner")

    else:
        raise ValueError(f"No output dataset type of {args.output_dataset_type}")

    ### Merge dataframes
    logger.info(
        "Annotating dataframes with tags for tracking which students have what data"
    )
    census_df[CensusDataColumns.has_census_data] = 1
    att_df[AttendanceDataColumns.has_attendance_data] = 1
    ks2_df[KSDataColumns.has_ks2_data] = 1
    neet_df[CCISDataColumns.has_ccis_data] = 1
    chars_df[CharacteristicsDataColumns.has_characteristics_data] = 1

    logger.info("Annotating ks2 dataframe with age 10")
    ks2_df[AGE] = 10

    if args.output_dataset_type == OutputDatasetTypes.prediction:
        logger.info(
            "Merging in priority order characteristics, census, attendance, ks2, "
        )
        merge_list = [chars_df, census_df, att_df, ks2_df]
    elif args.output_dataset_type in [
        OutputDatasetTypes.unknowns,
        OutputDatasetTypes.modeling,
    ]:
        logger.info(
            "Merging in priority order 1. ccis, 2. census, 3. attendance, 4. ks2"
        )
        merge_list = [neet_df, census_df, att_df, ks2_df]
    merged_df = mu.merge_priority_dfs(
        dfs=merge_list, on=UPN, how="outer", unknown_vals=UNKNOWN_CODES, na_vals=NA_VALS
    )
    #breakpoint()
    # If there is an NA value this means that row wasn't in the respective dataset, so fillna with 0.
    for col in [
        CensusDataColumns.has_census_data,
        AttendanceDataColumns.has_attendance_data,
        KSDataColumns.has_ks2_data,
        CCISDataColumns.has_ccis_data,
        CharacteristicsDataColumns.has_characteristics_data,
    ]:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)

    merged_df = d.downcast_df(merged_df, inplace=True)
    merged_df = merged_df.reset_index(drop=True)

    merged_df = merged_df.convert_dtypes()

    # Set to None so python garbage collector will come clean them up
    neet_df = None
    census_df = None
    att_df = None
    ks2_df = None
    chars_df = None

    logger.info(f"Initial merged row count {len(merged_df)}")
    logger.info(f"Initial column count {len(merged_df.columns)}")

    logger.info("Inferring age and school year")
    # This will infer as many years and ages as we can.
    merged_df[YEAR] = merged_df[YEAR].fillna(
        school_year_of_age(merged_df)
    )  # Get years from known ages, birth_month, birth_year
    for col in [
        YEAR,
        CharacteristicsDataColumns.birth_year,
        CharacteristicsDataColumns.birth_month,
    ]:
        merged_df[col] = merged_df[col].astype(pd.Int16Dtype())
    merged_df[AGE] = merged_df[AGE].fillna(
        nu.compute_age(merged_df)
    )  # Get ages from known years, birth_month, birth_year

    ### Final processing based on output dataset type
    if args.output_dataset_type == OutputDatasetTypes.modeling:
        # All data should have a birth year and birth month
        # All data should either have an age or year, so after
        # inferal, there should be no missing ages.
        assert not merged_df[AGE].isna().any()
        logger.info(f"Dropping samples with age >= 15")
        merged_df = merged_df[merged_df[CCISDataColumns.age] < 15]

    logger.info(f"Dropping birth year column")
    merged_df.drop(CharacteristicsDataColumns.birth_year, axis=1, inplace=True)
    if args.only_att_census_merge:
        logger.info(f"Dropping birth month column")
        merged_df.drop(CharacteristicsDataColumns.birth_month, axis=1, inplace=True)       

    logger.info("Dropping unnecessary columns")
    if args.output_dataset_type == OutputDatasetTypes.modeling:
        merged_df = d.drop_empty_cols(merged_df, na_vals=NA_VALS, logger=logger)
        merged_df = d.drop_single_valued_cols(merged_df, logger=logger)
    if args.output_dataset_type in (
        OutputDatasetTypes.unknowns,
        OutputDatasetTypes.prediction,
    ):
        logger.info(
            "Building an inference dataset so getting rid of any target columns and other ccis columns"
        )
        merged_df = d.safe_drop_columns(
            merged_df, columns=non_prediction_columns, inplace=True
        )
    merged_df = d.drop_duplicate_rows(merged_df, logger=logger)

    logger.info("Filling in missing values of columns that are already categorical")
    for col in merged_df.columns:
        if d.is_categorical(col):
            merged_df[col] = merged_df[col].fillna(0)

    logger.info(f"Final merged row count {len(merged_df)}")
    census_count = merged_df[CensusDataColumns.has_census_data].sum()
    ks_count = merged_df[KSDataColumns.has_ks2_data].sum()
    att_count = merged_df[AttendanceDataColumns.has_attendance_data].sum()
    logger.info(
        f"Final census count in merged {census_count}/{len(merged_df)} ({py.safe_divide(census_count, len(merged_df))})"
    )
    logger.info(
        f"Final KS count in merged {ks_count}/{len(merged_df)} ({py.safe_divide(ks_count, len(merged_df))})"
    )
    logger.info(
        f"Final att count in merged {att_count}/{len(merged_df)} ({py.safe_divide(att_count, len(merged_df))})"
    )
    if args.chars is not None:
        chars_count = merged_df[CharacteristicsDataColumns.has_characteristics_data].sum()
        logger.info(
            f"Final characteristics count in merged {chars_count}/{len(merged_df)} ({py.safe_divide(chars_count, len(merged_df))})"
        )
    logger.info(f"Final column count {len(merged_df.columns)}")
    
    #breakpoint()
    
    csv_fp = f.tmp_path(args.output, debug=args.debug)

    logger.info(f"Saving multi upn merged data to {csv_fp}")
    merged_df.to_csv(csv_fp, index=False)
