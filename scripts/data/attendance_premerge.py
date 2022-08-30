"""
Prepares the attendance dataset for merging with other datasets

This stage contains different options for how to aggregate the attendance data across terms for each student.

1. Exact: Sums over multiple entries for student per term
2. Percent1: Calculates attendance percentages for each student per year 
3. Percent2: Calculates attendance percentages for each student per year 

Parameters
----------
--debug: bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
--input: string (filepath)
    Filepath of the attendance csv file. This is a required parameter.
--output: string (filepath)
    Filepath of where to save the output csv file. This is a required parameter. 
--attendance_type:  string
    which function to use to determine how to aggregate the attendance data. This is a required parameter.

Returns
-------
csv file
    csv file of attendance data ready to be merged with other datasets

"""

import argparse
from dataclasses import asdict
import pandas as pd

# DVC Params
from src.constants import AttendanceDataColumns, NA_VALS, AttendanceTypes

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d
from src import attendance_utils as au
from src import py_utils as py

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input", type=lambda x: x.strip("'"),
    required=True, help="where to find input annotated attendance csv"
)
parser.add_argument(
    "--output", type=lambda x: x.strip("'"),
    required=True, help="where to output the produced csv file"
)
parser.add_argument(
    "--attendance_type",
    type=lambda x: x.strip("'"),
    required=True,
    choices=asdict(AttendanceTypes).values(),
    help="how to aggregate the attendance data across terms for each student-year",
)


def create_attendance_exact_df(df, logger=l.PrintLogger):
    """
    Groups attendance data by UPN and year to sum total number of absence days and termly sessions possible in each year for each student.

    Parameters
    -----------
    df: pd.DataFrame
        Dataframe containing attendance data with separate term data for each student
    logger:

    Returns
    -----------
    pd.DataFrame
        dataframe with attendance data aggregated so only one attendance row per year for each student

    """
    gby_columns = [AttendanceDataColumns.upn, AttendanceDataColumns.year]

    logger.info("Drop all columns that are not useful for attendance numbers")
    keep_columns = [c for c in df.columns if au.keep_column_criteria(c, gby_columns)]
    logger.info(f"Keeping columns {keep_columns}")
    df = df[keep_columns]

    logger.info("Do a groupby and sum")
    # The only columns left that are not na should all be numerical columns
    numerical_cols = [c for c in df.columns if c not in gby_columns]
    logger.info(f"Summing over columns {numerical_cols}")
    df.loc[:, numerical_cols] = df[numerical_cols].astype(pd.Int16Dtype())
    df = df[
        df["total_absences"] <= df["termly_sessions_possible"]
    ]  # keep rows only where total absences is not greater than termly sessions possible
    df = df.groupby(by=gby_columns).sum().reset_index()

    return df


def create_attendance_percent1_df(df, logger=l.PrintLogger):
    """
    1. Sums data across terms for each student
    2. Computes percentage of total absences wrt termly sessions possible
    3. Computes percentage of each absence reason wrt total absences
    4. Computes percentage of late_present wrt termly_sessions_possible
    5. Computes percentage of total_nonabsences wrt termly_sessions_possible + total_nonabsences
    6. Computes percentage of each nonabsence reason column wrt total_nonabsences
    7. Drops the termly sessions possible column

    Parameters
    -----------
    df: pd.DataFrame
        Dataframe containing attendance data with separate term data for each student
    logger


    Returns
    ----------
    pd.DataFrame
        dataframe with attendance aggregated so only one attendance row per year for each student. The data is represented as percentages.

    """
    df = create_attendance_exact_df(df, logger=logger)

    # breakpoint()

    absence_columns = au.get_absence_reason_columns(df)
    nonabsence_columns = au.get_nonabsence_reason_columns(df)

    logger.info("Compute percentage of sessions possible the student attended")
    df_percent = df.copy()

    df_percent[au.AttendanceDataColumns.total_absences] = d.divide_with_na(
        df[au.AttendanceDataColumns.total_absences],
        df[au.AttendanceDataColumns.termly_sessions_possible],
        fillna=0,
        fill_prev_na=False,
    )

    logger.info(f"Computing percentage of cols {absence_columns} wrt total_absences")
    for col in absence_columns:
        # Percent of total absences
        df_percent[col] = d.divide_with_na(
            df[col],
            df[au.AttendanceDataColumns.total_absences],
            fillna=0,
            fill_prev_na=False,
        )

    # Percent late of sessions possible
    logger.info(f"Computing percentage of late_present wrt termly_sessions_possible")
    df_percent[au.AttendanceDataColumns.late_present] = d.divide_with_na(
        df[au.AttendanceDataColumns.late_present],
        df[au.AttendanceDataColumns.termly_sessions_possible],
        fillna=0,
        fill_prev_na=False,
    )

    # Percent of total sessions spent in nonabsence
    logger.info(
        f"Computing percentage of total_nonabsences wrt termly_sessions_possible + total_nonabsences"
    )
    df_percent[au.AttendanceDataColumns.total_nonabsences] = d.divide_with_na(
        df[au.AttendanceDataColumns.total_nonabsences],
        df[au.AttendanceDataColumns.total_nonabsences]
        + df[au.AttendanceDataColumns.termly_sessions_possible],
        fillna=0,
        fill_prev_na=False,
    )

    logger.info(
        f"Computing percentage of cols {nonabsence_columns} wrt total_nonabsences"
    )
    for col in nonabsence_columns:
        df_percent[col] = d.divide_with_na(
            df[col],
            df[au.AttendanceDataColumns.total_nonabsences],
            fillna=0,
            fill_prev_na=False,
        )

    logger.info(f"Dropping the termly_sessions_possible column")
    df_percent = df_percent.drop(
        au.AttendanceDataColumns.termly_sessions_possible, axis=1
    )

    return df_percent


def create_attendance_percent2_df(df, logger=l.PrintLogger):
    """
    1. Sums data across terms for each student
    2. Computes percentage of total absences wrt termly sessions possible
    3. Computes percentage of sessions spent in nonabsence wrt termly sessions possible
    4. Drops the termly sessions possible column

    Parameters
    -----------
    df: pd.DataFrame
        Dataframe containing attendance data with separate term data for each student
    logger

    Returns
    ----------
    pd.DataFrame
        dataframe with attendance aggregated so only one attendance row per year for each student. The data is represented as percentages.
    """

    df = create_attendance_exact_df(df, logger=logger)

    absence_columns = au.get_absence_reason_columns(df)
    nonabsence_columns = au.get_nonabsence_reason_columns(df)

    logger.info("Compute percentage of sessions possible the student attended")
    df_percent = df.copy()
    by_session_cols = [
        AttendanceDataColumns.total_absences,
        AttendanceDataColumns.late_present,
    ] + absence_columns
    logger.info(
        f"Computing percentage for cols {by_session_cols} wrt termly_sessions_possible"
    )
    for col in by_session_cols:
        df_percent[col] = d.divide_with_na(
            df_percent[col],
            df[au.AttendanceDataColumns.termly_sessions_possible],
            fillna=0,
            fill_prev_na=False,
        )
    # Percent of total sessions spent in nonabsence
    by_total_sessions_cols = [
        AttendanceDataColumns.total_nonabsences
    ] + nonabsence_columns
    logger.info(
        f"Computing percentage for cols {by_total_sessions_cols} wrt termly_sessions_possible + total_nonabsences"
    )
    for col in by_total_sessions_cols:
        df_percent[col] = d.divide_with_na(
            df_percent[col],
            df[au.AttendanceDataColumns.total_nonabsences]
            + df[au.AttendanceDataColumns.termly_sessions_possible],
            fillna=0,
            fill_prev_na=False,
        )

    logger.info(f"Dropping the termly_sessions_possible column")
    df_percent = df_percent.drop(
        au.AttendanceDataColumns.termly_sessions_possible, axis=1
    )

    return df_percent


def create_attendance_normed_df(df, eps=1e-9, logger=l.PrintLogger):
    """
    1. Compute mean and std for each term
    2. Normalize each students values per term
    3. Aggregate terms by averaging
    """
    gby_columns = [
        AttendanceDataColumns.upn,
        AttendanceDataColumns.year,
        AttendanceDataColumns.term_type,
    ]
    norm_gby_columns = [AttendanceDataColumns.year, AttendanceDataColumns.term_type]
    final_gby_columns = [AttendanceDataColumns.upn, AttendanceDataColumns.year]

    logger.info("Keep all columns that are useful for attendance numbers")
    keep_columns = [c for c in df.columns if au.keep_column_criteria(c, gby_columns)]
    logger.info(f"Keeping {keep_columns}")
    df = df[keep_columns]

    # The only columns left that are not na should all be numerical columns
    logger.info("Do a groupby and sum")
    numerical_cols = [c for c in df.columns if c not in gby_columns]
    logger.info(f"Summing over columns {numerical_cols}")
    df.loc[:, numerical_cols] = d.to_int(df[numerical_cols])
    df = df.groupby(by=gby_columns).sum().reset_index()

    logger.info("Normalize each column on year and term")
    # Mean and std will only use numeric columns
    df_term_means = df.groupby(by=norm_gby_columns).mean().reset_index()
    df_term_stds = df.groupby(by=norm_gby_columns).std().reset_index()
    df_with_norms = df.merge(
        df_term_means, on=norm_gby_columns, suffixes=(None, "_mean"), how="left"
    ).merge(df_term_stds, on=norm_gby_columns, suffixes=(None, "_std"), how="left")

    df_means = df_with_norms[[c for c in df_with_norms.columns if c.endswith("_mean")]]
    df_means.columns = df_means.columns.map(lambda x: py.remove_suffix(x, "_mean"))
    df_stds = df_with_norms[[c for c in df_with_norms.columns if c.endswith("_std")]]
    df_stds.columns = df_stds.columns.map(lambda x: py.remove_suffix(x, "_std"))

    # TODO: Check if d.divide_with_na here fixes the issue of many Nans cropping up
    df_normed = df.copy()
    for col in df_means.columns:
        df_normed[col] = d.divide_with_na(
            df_normed[col] - df_means[col], df_stds[col], fillna=0, fill_prev_na=False
        )

    logger.info("Average over terms in year")
    df_normed = df_normed.groupby(by=final_gby_columns).mean().reset_index()

    return df_normed


if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    df = d.load_csv(
        args.input,
        drop_empty=True,  # This is a model dataset, so drop all irrelevant features
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        use_na=True,
        convert_dtypes=True,
        na_vals=NA_VALS,
    )

    # Drop termly sessions authorised, termly sessions unauthorised, present_am, present_pm columns
    logger.info("Dropping inconsistent columns")
    drop_cols = [
        AttendanceDataColumns.termly_sessions_authorised,
        AttendanceDataColumns.termly_sessions_unauthorised,
        AttendanceDataColumns.present_am,
        AttendanceDataColumns.present_pm,
    ]
    df = d.safe_drop_columns(df, drop_cols)

    logger.info(f'Creating attendance dataset of type "{args.attendance_type}".')
    if args.attendance_type == AttendanceTypes.exact:
        df = create_attendance_exact_df(df, logger)
    elif args.attendance_type == AttendanceTypes.percent1:
        df = create_attendance_percent1_df(df, logger)
    elif args.attendance_type == AttendanceTypes.percent2:
        df = create_attendance_percent2_df(df, logger)
    elif args.attendance_type == AttendanceTypes.term_normalized:
        df = create_attendance_normed_df(df, logger=logger)
    else:
        raise NotImplementedError(
            f'Attendance type "{args.attendance_type}" is not valid'
        )

    csv_fp = f.tmp_path(args.output, debug=args.debug)

    logger.info(f"Saving attendance premerge data to {csv_fp}")
    df.to_csv(csv_fp, index=False)
