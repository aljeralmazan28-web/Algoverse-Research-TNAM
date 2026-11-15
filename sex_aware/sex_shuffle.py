"""
sex_shuffle.py -- shuffled-sex-label control (task 4). numpy only, no torch.

Randomly permutes the sex labels that the MODEL sees (sex input + reweighting).
Per-sex evaluation must still use the TRUE labels; only model inputs are shuffled.
"""
import numpy as np


def shuffle_sex_labels(sex, tissue=None, within_tissue=True, seed=0):
    """
    sex           : array of labels, one per cell line (NOT per line-drug row)
    tissue        : array of tissue names, one per cell line (needed if within_tissue)
    within_tissue : True  -> permute inside each tissue, so each tissue keeps its
                             exact number of F and M lines (tissue structure intact)
                    False -> permute across all lines (global shuffle)
    Returns a new array; the input is not modified.

    Shuffle at the CELL-LINE level, then map to rows. Shuffling per row would give
    one line different "sexes" for different drugs, which is not the control we want.
    """
    sex = np.asarray(sex)
    rng = np.random.default_rng(seed)
    out = sex.copy()
    if not within_tissue:
        out[:] = sex[rng.permutation(len(sex))]
        return out
    if tissue is None:
        raise ValueError("tissue is required when within_tissue=True")
    tissue = np.asarray(tissue)
    for t in np.unique(tissue):
        idx = np.where(tissue == t)[0]
        out[idx] = sex[idx[rng.permutation(len(idx))]]
    return out