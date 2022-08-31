"""
Canonicalizes dataset by applying the following operations
1. Rename columns to use snake_case. We rename coded column names to their full names.
2. If the dataset_type is school_info, we add an extra row for the 7777 establishment number

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
inputs : string(s)
    The filepaths to the csvs. 
outputs : string(s)
    The filepaths to the output canonicalized csvs. 
dataset_type : {attendance, census, ccis, ks4, school_info, ks2, characteristics}
    The type of dataset to canonicalize. This will determine what column renaming
    will be done.

Returns
---------
csv file
   canonicalized attendance dataset saved at output filepath as a csv file.

"""
from dataclasses import asdict
import pandas as pd
import argparse

# DVC Params
from src.constants import DatasetTypes, RENAME_DICT, SchoolInfoColumns, NA_VALS

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d


parser = argparse.ArgumentParser(description="")
parser.add_argument("--debug", action="store_true", help="run transform in debug mode")
parser.add_argument(
    "--inputs", required=True, nargs="+", help="where to find the input attendance csvs"
)
parser.add_argument(
    "--outputs",
    required=True,
    nargs="+",
    help="where to put the output canonicalized attendance csvs",
)
parser.add_argument(
    "--dataset_type",
    required=True,
    choices=asdict(DatasetTypes).values(),
    help="what type of dataset each of the inputs is.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.dataset_type in [DatasetTypes.characteristics, DatasetTypes.ks2]:
        raise NotImplementedError(
            "We have not implemented canonicalization for dataset of characteristics or of ks2."
        )

    if len(args.inputs) != len(args.outputs):
        raise ValueError(
            f"The number of input files must match the number of output files. You had {len(args.inputs)} input files and {len(args.outputs)} output files. The inputs were {args.inputs} and the outputs were {args.outputs}."
        )

    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)

    # Drop duplicate rows
    for input_fp, output_fp in zip(args.inputs, args.outputs):
        df: pd.DataFrame = d.load_csv(
            input_fp,
            drop_empty=False,
            drop_single_valued=False,
            drop_duplicates=False,  # Canonicalize is non-destructive,
            read_as_str=True,
            use_na=True,
            na_vals=NA_VALS,
            logger=logger,
        )

        # Do column name rename
        rename = RENAME_DICT[args.dataset_type]
        logger.info(f"Renaming columns of {args.dataset_type} dataset")
        df.rename(columns=rename, inplace=True)

        # Do school info specific logic
        if args.dataset_type == DatasetTypes.school_info:
            # Add the special school
            new_row = pd.DataFrame(
                [
                    {
                        SchoolInfoColumns.la_establishment_number: "8257777",
                        SchoolInfoColumns.establishment_number: "7777",
                        SchoolInfoColumns.establishment_name: "Other Special",
                        SchoolInfoColumns.establishment_type: "Special",
                        SchoolInfoColumns.establishment_status: "",
                        SchoolInfoColumns.establishment_area: "",
                        SchoolInfoColumns.establishment_postcode: "",
                        SchoolInfoColumns.establishment_electoral_wards: "",
                    }
                ]
            )
            assert set(new_row.keys()) == set(df.columns)
            df = pd.concat([df, new_row], axis=0).reset_index(drop=True)
            # Parse the list of electoral wards
            #breakpoint()
            df[SchoolInfoColumns.establishment_electoral_wards] = df[
                SchoolInfoColumns.establishment_electoral_wards
            ].apply(lambda x: d.parse_human_list(x))

            # Remove the "- The" at the end of some school names
            #breakpoint()
            df[SchoolInfoColumns.establishment_name] = df[
                SchoolInfoColumns.establishment_name
            ].apply(lambda x: x.replace(" - The", "") if not pd.isna(x) else x)

        csv_fp = f.tmp_path(output_fp, debug=args.debug)
        logger.info(f"Saving canonicalized data to {csv_fp}")
        df.to_csv(csv_fp, index=False)
