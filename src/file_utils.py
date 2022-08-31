"""
This utils file contains helpers for accessing filepaths.
"""
from datetime import datetime
from typing import Dict
import os

from .params import TMP_DIR, INTERIM_MODELS_DIR, METRICS_DIR

# File utils
def safe_open_dir(dirpath: str) -> str:
    if not os.path.isdir(dirpath):
        print(f"Directory {dirpath} does not exist, creating it")
        os.makedirs(dirpath)
    return dirpath


def safe_open_file(fp: str) -> str:
    dirname, basename = os.path.split(fp)
    dirname = safe_open_dir(dirname)
    return fp


def tmp_path(fp: str, debug=True):
    return (
        os.path.join(safe_open_dir(TMP_DIR), os.path.basename(fp))
        if debug
        else safe_open_file(fp)
    )


def tmp_paths(path_dict: Dict[str, str], debug=True):
    return {k: tmp_path(fp, debug=debug) for k, fp in path_dict.items()}


def get_canonical_filename(fp):
    return os.path.splitext(os.path.basename(fp))[0]


def tag_file(fp, **tags):
    dirname, basename = os.path.split(fp)
    canonical_filename, ext = os.path.splitext(basename)
    tag_strs = ["_".join(item) for item in tags.items()]
    new_canonical_filename = "_".join([canonical_filename] + tag_strs)

    return os.path.join(dirname, new_canonical_filename + ext)


def get_checkpoint_filepath(space, dataset_suffix):
    # timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    return os.path.join(INTERIM_MODELS_DIR, f"{space}_{dataset_suffix}.pkl")


def get_cv_results_filepath(space, dataset_suffix):
    return os.path.join(METRICS_DIR, f"{space}_{dataset_suffix}.csv")
