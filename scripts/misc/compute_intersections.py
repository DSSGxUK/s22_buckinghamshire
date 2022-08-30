import os
import pandas as pd
import numpy as np
from src import constants, data_utils
from tqdm import trange

DEBUG = False
CHUNK_SIZE = int(1000)


def extract_ind_dfs(merged_df, year_col):
    years = merged_df[year_col].unique()
    return {d: merged_df[merged_df[year_col] == d] for d in years}


def log_dfs_output(dfs, col):
    return [d[col].head() for d in dfs]


def load_dfs(list_fps):
    return []


def flatten_dict(d):
    new_d = {}
    for k, v in d.items():
        for k2, v2 in v.items():
            new_d["_".join([str(k), str(k2)])] = v2
    return new_d


all_merged_dfs = {
    k: pd.read_csv(fp, keep_default_na=False, dtype=str)
    for k, fp in constants.MERGED_FPS.items()
}
all_dfs = flatten_dict(
    {
        k: extract_ind_dfs(mdf, constants.YEAR_COLS[k])
        for k, mdf in all_merged_dfs.items()
    },
)
if DEBUG:
    all_dfs = {k: v for k, v in list(all_dfs.items())[:13]}

all_upns = {
    k: data_utils.get_unique_nonna(d, constants.UPN) for k, d in all_dfs.items()
}

# Now do all combinations
def enumerate_binary_inputs(window: int):
    return dec2bitarray(np.arange(2**window), window)


def dec2bitarray(arr, num_bits: int, little_endian: bool = False):
    if little_endian:
        shift_arr = np.arange(num_bits)
    else:
        shift_arr = np.arange(num_bits)[::-1]
    return np.right_shift(np.expand_dims(arr, -1), shift_arr) % 2


data_names = np.array(list(all_upns.keys()))
# data_names
num_datasets = len(data_names)
combo_masks = enumerate_binary_inputs(num_datasets)[1:].astype(bool)

data_names = np.array(list(all_upns.keys()))
every_upn = sorted(list(set.union(*all_upns.values())))
bin_masks = []
for n, ds in all_upns.items():
    bin_mask = [1 if u in all_upns[n] else 0 for u in every_upn]
    bin_masks.append(bin_mask)

bin_masks = np.array(bin_masks)  # rows are names, cols are upns
bin_masks = bin_masks.astype(bool)

# bin_masks are Names x UPNs
# combo_masks are Masks x Names
# Masks x Names x 1 * 1 x Names x UPNs

# Masks x UPNs
print("Computing combo masks")
combo_unique_counts_list = []
for chunk in trange(0, combo_masks.shape[0], CHUNK_SIZE):
    combo_uniques = np.all(
        combo_masks[chunk : chunk + CHUNK_SIZE, ..., None] & bin_masks[None, ...]
        | ~combo_masks[chunk : chunk + CHUNK_SIZE, ..., None],
        axis=1,
    )
    # Masks
    combo_unique_counts_list.append(np.sum(combo_uniques.astype(np.int32), axis=1))
combo_unique_counts = np.concatenate(combo_unique_counts_list, axis=0)

print("Creating counts csv")
mask_names = [" x ".join(data_names[combo_mask]) for combo_mask in combo_masks]
df = pd.DataFrame(
    combo_unique_counts[:, None], index=mask_names, columns=["shared_upns"]
)
df.to_csv(os.path.join(constants.RESULTS_DIR, "shared_upns.csv"))
