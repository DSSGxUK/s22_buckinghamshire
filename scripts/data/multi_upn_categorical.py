"""
This stage converts multi-upn dataset columns containing categorical data to multiple categorical columns containing 0 and 1. 
e.g. the original gender column contained 4 values [M, F, U, W]
This column will be converted to 4 new columns [gender_M, gender_F, gender_U, gender_W] containing 0 or 1.   

Performing the following transformations
# To Categorical
1. enrol_status
2. establishment_status
3. establishment_type
4. ethnicity
5. gender
6. language
7. resourced_provision_indicator
  - different cols for 1 or 0, with both 0 if value is na
8. sen_need1 + sen_need2
9. sen_unit_indicator
  - different cols for True or False, with both 0 if value is na
10. sen_provision

Also includes a function for splitting columns with mixed (numeric and categorical) data (mostly occurs in some of the ks2 data columns)

Parameters
----------
--debug: bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
--input: string (filepath)
    Filepath of the input csv file. This is a required parameter.
--output: string (filepath)
    Filepath of where to save the output csv file. This is a required parameter. 
--include_test_taken_code: bool
    Whether to include a code that tells us the student took the ks2 test.

Returns
-------
csv file
   multi upn categorical dataset saved at output filepath as a csv file.  
"""

import pandas as pd
import argparse
from dataclasses import asdict

# DVC Params
from src.constants import (
    CCISDataColumns,
    CensusDataColumns,
    SchoolInfoColumns,
    KSDataColumns,
    TEST_TAKEN_CODE,
    UPN,
    NA_VALS,
    CATEGORICAL_SEP,
    OutputDatasetTypes,
    CharacteristicsDataColumns
)

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input", type=lambda x: x.strip("'"), required=True, help="where to find merged multi upn dataset"
)
parser.add_argument(
    "--output", type=lambda x: x.strip("'"), required=True, help="where to put merged multi upn categorical dataset"
)
parser.add_argument(
    "--include_test_taken_code",
    action="store_true",
    help="whether to include a code that tells us the student took the ks2 test.",
)
parser.add_argument(
    "--output_dataset_type",
    type=lambda x: x.strip("'"), required=True,
    choices=asdict(OutputDatasetTypes).values(),
    help="What kind of output merged dataset to build",
)

def demix_column(
    mixed_data, numeric_name, categorical_name, include_test_taken_code: bool
):
    """
    Splits columns with mixed numeric and categroical data into 2 columns

    Parameters
    ----------
    mixed_data: pd.DataFrame column
        Column with mixed numeric and categorical data that requires de-mixing
    numeric_name: string
        Name of the column that requires de-mixing. This column will also contain the numeric data after de-mixing.
    categorical_name: string
        Name of the new column that will contain the categorical data. Name is the `numeric_name` with `_codes` added as suffix.
    include_test_taken_code: bool
        Whether to include a code that tells us the student took the ks2 test.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 2 columns with separated numeric and categorical data. If `include_test_taken_code` is True a code is included that tells us the student took the ks2 test.

    """

    # To deal with mixed columns, let's split into two columns, one with numbers and another with codes
    where_numeric = d.is_number(mixed_data)
    numeric_data = mixed_data[where_numeric]
    numeric_data.name = numeric_name

    categorical_data = mixed_data[~where_numeric].astype(
        pd.StringDtype()
    )  # Categorical data will be treated as strings
    categorical_data.name = categorical_name
    demixed = pd.concat([numeric_data, categorical_data], axis=1).convert_dtypes()
    if include_test_taken_code:
        demixed.loc[where_numeric, categorical_data.name] = TEST_TAKEN_CODE
    return demixed


if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    logger.info(f"Building a {args.output_dataset_type} dataset.")

    df = d.load_csv(  # All unnecessary columns and rows should already be dropped.
        args.input,
        drop_empty=False,
        drop_single_valued=False,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=False,
        use_na=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        convert_dtypes=True,
        logger=logger,
    )

    logger.info(f"Initial merged row count {len(df)}")
    logger.info(f"Initial column count {len(df.columns)}")

    # Census Data, to categorical
    simple_categorical_columns = [
        CensusDataColumns.enrol_status,
        SchoolInfoColumns.establishment_status,
        SchoolInfoColumns.establishment_type,
        CensusDataColumns.ethnicity,
        CensusDataColumns.gender,
        CensusDataColumns.language,
        CensusDataColumns.resourced_provision_indicator,
        CensusDataColumns.sen_unit_indicator,
        CensusDataColumns.sen_provision,
    ]
    df = d.get_dummies_with_logging(
        df, columns=simple_categorical_columns, logger=logger
    )

    logger.info(f"Converting census sen need into a single categorical variable")
    sen_need_values = sorted(
        set.union(
            set(d.dropna(df[CensusDataColumns.sen_need1])),
            set(d.dropna(df[CensusDataColumns.sen_need2])),
        )
    )
    logger.info(f"Found sen need values {sen_need_values}")
    df[CensusDataColumns.sen_need1] = pd.Categorical(
        df[CensusDataColumns.sen_need1], categories=sen_need_values
    )
    df[CensusDataColumns.sen_need2] = pd.Categorical(
        df[CensusDataColumns.sen_need2], categories=sen_need_values
    )
    dummy_sen_need = pd.concat(
        [
            pd.get_dummies(df[CensusDataColumns.sen_need1]),
            pd.get_dummies(df[CensusDataColumns.sen_need2]),
        ]
    ).max(level=0)
    dummy_sen_need = dummy_sen_need.add_prefix("sen_need" + CATEGORICAL_SEP)
    df.drop(
        [CensusDataColumns.sen_need1, CensusDataColumns.sen_need2], axis=1, inplace=True
    )
    df = pd.concat([df, dummy_sen_need], axis=1)

    # KS2
    mix_columns = [
        KSDataColumns.ks2_english,
        KSDataColumns.ks2_mathematics,
        KSDataColumns.ks2_english_ta,
        KSDataColumns.ks2_english_ta_level,
        KSDataColumns.ks2_english_level_finely_graded,
        KSDataColumns.ks2_mathematics_ta,
        KSDataColumns.ks2_mathematics_ta_level,
        KSDataColumns.ks2_mathematics_level_finely_graded,
        KSDataColumns.ks2_reading_ta_level,
        KSDataColumns.ks2_reading_level_finely_graded,
        KSDataColumns.ks2_mathematics_finely_graded,
    ]
    code_columns = [n + "_codes" for n in mix_columns]
    logger.info(
        f"Demixing KS2 columns {mix_columns} with both numeric and categorical data"
    )
    logger.info(
        f'We are {"" if args.include_test_taken_code else "not "} including a code for whether the test was taken in the categorical codes columns'
    )

    for col_name, code_col_name in zip(mix_columns, code_columns):
        demixed_df = demix_column(
            df[col_name],
            numeric_name=col_name,
            categorical_name=code_col_name,
            include_test_taken_code=args.include_test_taken_code,
        )
        df.drop(col_name, axis=1, inplace=True)
        df = pd.concat([df, demixed_df], axis=1)

    simple_categorical = [
        KSDataColumns.sen,
    ] + code_columns
    logger.info(
        f"Creating simple categorical columns (using pd.get_dummies) for KS2 columns {simple_categorical}"
    )
    df = d.get_dummies_with_logging(df, columns=simple_categorical, logger=logger)

    
    if args.output_dataset_type == OutputDatasetTypes.prediction:
        # Characteristics
        simple_categorical_columns = [
            CharacteristicsDataColumns.birth_month,
            CCISDataColumns.characteristic_code,
            CCISDataColumns.level_of_need_code,
            CCISDataColumns.send_flag,
            CCISDataColumns.sen_support_flag,
        ]
        df = d.get_dummies_with_logging(
            df, columns=simple_categorical_columns, logger=logger
        )
    elif args.output_dataset_type in [OutputDatasetTypes.modeling, OutputDatasetTypes.unknowns]:
        # CCIS
        simple_categorical_columns = [
            CCISDataColumns.birth_month,
        ]
        df = d.get_dummies_with_logging(
            df, columns=simple_categorical_columns, logger=logger
        )
    else:
        raise NotImplementedError(f"Dataset type {args.output_dataset_type} is not implemented.")

    logger.info(f"Lowering all capital letters in column names")
    df.columns = df.columns.map(str.lower)

    logger.info(f"Final merged row count {len(df)}")
    logger.info(f"Final column count {len(df.columns)}")

    csv_fp = f.tmp_path(args.output, debug=args.debug)

    logger.info(f"Saving categorical data to {csv_fp}")
    df.to_csv(csv_fp, index=False)
