"""
Prepares census dataset prior to merge with other datasets.

Performing the following transformations
# Drop columns
2. entry_date
3. establishment_area
4. establishment_electoral_wards
5. establishment_name
6. establishment_number
7. establishment_postcode
8. la_establishment_number

# Convert column to binary values:
1. fsme_on_census_day

Parameters
----------
--debug: bool
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
--input: string (filepath)
    Filepath of the census csv file. This is a required parameter.
--output: string (filepath)
    Filepath of where to save the output csv file. This is a required parameter. 

Returns
-------
csv file
    csv file of census data ready to be merged with other datasets

"""
import argparse

# DVC Params
from src.constants import (
    CensusDataColumns,
    SchoolInfoColumns,
    NA_VALS,
    UPN,
)

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--input", type=lambda x: x.strip("'"),
    required=True, help="where to find input annotated census csv"
)
parser.add_argument(
    "--output", type=lambda x: x.strip("'"),
    required=True, help="where to output the produced csv file"
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    df = d.load_csv(  # Modeling dataset so we drop unnecessary columns and entries
        args.input,
        drop_empty=True,
        drop_single_valued=True,
        drop_duplicates=True,
        read_as_str=False,
        drop_missing_upns=True,
        upn_col=UPN,
        na_vals=NA_VALS,
        logger=logger,
    )

    #     breakpoint()

    logger.info(f"Initial row count {len(df)}")
    logger.info(f"Initial column count {len(df.columns)}")

    # Drop
    modeling_columns = [
        SchoolInfoColumns.establishment_type,
        SchoolInfoColumns.establishment_status,
        CensusDataColumns.upn,
        CensusDataColumns.gender,
        CensusDataColumns.enrol_status,
        CensusDataColumns.ethnicity,
        CensusDataColumns.language,
        CensusDataColumns.sen_provision,
        CensusDataColumns.sen_need1,
        CensusDataColumns.sen_need2,
        CensusDataColumns.sen_unit_indicator,
        CensusDataColumns.resourced_provision_indicator,
        CensusDataColumns.fsme_on_census_day,
        CensusDataColumns.age,
        CensusDataColumns.year,
    ]
    logger.info(f"Keeping the modeling columns {modeling_columns}")
    df = df[modeling_columns]

    logger.info(f"Final row count {len(df)}")
    logger.info(f"Final column count {len(df.columns)}")

    csv_fp = f.tmp_path(args.output, debug=args.debug)

    logger.info(f"Saving census premerge data to {csv_fp}")
    df.to_csv(csv_fp, index=False)
