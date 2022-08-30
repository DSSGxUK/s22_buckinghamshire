"""
This stage separates the multi-upn dataset into pre- and post-covid datasets.
We do this to separate students whose schooling and neet status may have been impacted by covid so we can train the models separately with these 2 groups of students.  
Pre-covid dataset includes students that were age 14 prior to 2018. 
Post-covid dataset includes students that were age 14 in 2018 or later.   

Parameters
----------
--debug: bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
--input: string (filepath)
    Filepath of the input csv file. This is a required parameter.
--output_pre_covid: string (filepath)
    Filepath of where to save the pre-covid output csv file. This is a required parameter. 
--output_post_covid: string (filepath)
    Filepath of where to save the post-covid output csv file. This is a required parameter. 

Returns
-------
csv file
   Pre-covid dataset saved at `--output_pre_covid` filepath as a csv file.  
csv file
   Pre-covid dataset saved at `--output_post_covid` filepath as a csv file.  

"""

import pandas as pd
import argparse
from functools import reduce

# DVC Params
from src.constants import UPN, NA_VALS, AGE, YEAR, YEAR_OF_COVID_SPLIT

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d


def split_data(df, YEAR_OF_COVID_SPLIT, logger=l.PrintLogger):
    """
    This function takes the multiple UPN dataframe (with multiple years of data per student)
    and separates the students into two dataframes based on the year they turned 14 and therefore whether their schooling/neet status could have been impacted by covid. The default year is 2018 which means students that turned 14 in or after 2018 are added to the post-covid dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        This is the dataframe with multiple years of data per student to be split.
    logger: logging.logger (or something that implements the same logging methods)
        This is the python logger to use for logging progress. By default,
        this is the `PrintLogger` defined in `log_utils`.
    YEAR_OF_COVID_SPLIT: integer (year)
        The year to use to split the multi-upn dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe containing students that were not impacted by covid (turned 14 before 2018)
    pd.DataFrame
        Dataframe containing students that were impacted by covid (turned 14 in or after 2018)
    """

    df = df.copy()

    year_age_difference = YEAR_OF_COVID_SPLIT - 14

    logger.info(f"Splitting data into pre and post covid years")

    upns = []
    df_max_age_index = df.groupby(UPN).agg(
        {AGE: "idxmax"}
    )  # takes index of max age for each student (should be mostly 14)
    df_max_age_rows = df.iloc[
        df_max_age_index[AGE].values
    ]  # gets rows of max age for each student
    for val in df_max_age_rows[AGE].unique():  # loop through ages
        covid_years = df_max_age_rows[
            (df_max_age_rows[AGE] == val)
            & (df_max_age_rows[YEAR] >= (val + (year_age_difference)))
        ]  # identify students affected by covid (if students are 14 in or after 2018)
        upns.append(covid_years[UPN])  # get upns of these students

    df_all_upns = reduce(
        lambda df1, df2: pd.merge(df1, df2, how="outer", on=UPN), upns
    )  # merge all upns into one dataframe

    post_df = df[
        df[UPN].isin(df_all_upns[UPN])
    ]  # select all students that were affected by covid
    pre_df = df[
        ~df[UPN].isin(df_all_upns[UPN])
    ]  # select all students that were not affected by covid

    # breakpoint()

    return pre_df, post_df


parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input", required=True, help="where to find multi upn categorical dataset"
)
parser.add_argument(
    "--output_pre_covid", required=True, help="where to put pre-covid multi upn dataset"
)
parser.add_argument(
    "--output_post_covid",
    required=True,
    help="where to put post-covid multi upn dataset",
)

if __name__ == "__main__":
    args = parser.parse_args()

    CATEGORICAL_FP = args.input
    csv_fp_pre = args.output_pre_covid
    csv_fp_post = args.output_post_covid

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    df = d.load_csv(  # Modeling dataset so we drop unnecessary columns and entries
        CATEGORICAL_FP,
        drop_empty=True,
        drop_single_valued=True,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        use_na=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger,
    )

    # breakpoint()

    pre_df, post_df = split_data(df, YEAR_OF_COVID_SPLIT, logger=l.PrintLogger)

    # breakpoint()

    if args.debug:
        csv_fp_pre = f.tmp_path(csv_fp_pre)
        csv_fp_post = f.tmp_path(csv_fp_post)

    logger.info(f"Saving pre and post covid datasets to {csv_fp_pre} and {csv_fp_post}")
    pre_df.to_csv(csv_fp_pre, index=False)
    post_df.to_csv(csv_fp_post, index=False)
