"""
Convert excel files to a csv. Assumes that
all excel files are a single sheet.

Parameters
----------
debug : bool     
    If passed as an argument, it will run this pipeline stage in
    debug mode. This will lower the logging level to `DEBUG` and
    save all outputs into the `tmp` directory.
input : string(s)
    The filepaths to the data excel files. 
output : string(s)
    The filepath to the output excel files. 

Outputs
---------
csv file for each input excel file.
"""

import argparse

# Other code
from src import file_utils as f
from src import log_utils as l
from src import data_utils as d

parser = argparse.ArgumentParser(description='')
parser.add_argument('--debug', action='store_true',
                    help='run transform in debug mode')
parser.add_argument('--inputs', required=True, nargs="+",
                    help='where to find the input old attendance data excel files')
parser.add_argument('--outputs', required=True, nargs="+",
                    help='where to put the output attendance data csvs')


if __name__ == "__main__":
    args = parser.parse_args()
    
    if len(args.inputs) != len(args.outputs):
        raise ValueError(f"The number of input files must match the number of output files. You had {len(args.inputs)} input files and {len(args.outputs)} output files. The inputs were {args.inputs} and the outputs were {args.outputs}.")


    # Set up logging
    logger = l.get_logger(name=f.get_canonical_filename(__file__), debug=args.debug)
    
    for input_fp, output_fp in zip(args.inputs, args.outputs):
        csv_fp = f.tmp_path(output_fp, debug=args.debug)
    
        d.save_xl_to_csv(
            xl_fp=input_fp,
            csv_fp=csv_fp,
            logger=logger
        )
    