"""Sex-balanced sample weights by sex x tissue.

Usage:
    table = fit_sex_tissue_weights(train_lines)          # fit on TRAINING lines only
    w = apply_weights(train_rows, table)                 # rows can be (cell line, drug) pairs
    loss = (w * per_sample_loss).mean()                  # or sklearn: fit(..., sample_weight=w)
"""
import numpy as np
import pandas as pd

SEXES = ("Female", "Male")


def fit_sex_tissue_weights(lines, sex_col="final_sex", tissue_col="OncotreeLineage",
                           mode="within_tissue", max_weight=10.0):
    """Compute a weight for each (tissue, sex) group from training cell lines.

    lines: one row per cell line (not per drug pair).
    mode:
      "within_tissue": in each tissue with both sexes, each sex gets equal total
                       weight; the tissue's overall weight is unchanged (default).
      "joint":         every (tissue, sex) group gets equal total weight
                       (also rebalances tissues; more aggressive).
    Lines with Unknown sex and single-sex tissues keep weight 1 before normalizing.
    Weights are capped at max_weight, then scaled so the mean weight per line is 1.
    """
    df = lines[[tissue_col, sex_col]].dropna()
    counts = df.groupby([tissue_col, sex_col]).size().rename("n").reset_index()
    counts["weight"] = 1.0
    known = counts[sex_col].isin(SEXES)

    if mode == "within_tissue":
        for _, g in counts[known].groupby(tissue_col):
            if len(g) < 2:                                   # only one sex present
                continue
            counts.loc[g.index, "weight"] = (g["n"].sum() / len(g)) / g["n"]
    elif mode == "joint":
        n_groups, n_lines = known.sum(), counts.loc[known, "n"].sum()
        counts.loc[known, "weight"] = n_lines / (n_groups * counts.loc[known, "n"])
    else:
        raise ValueError("mode must be 'within_tissue' or 'joint'")

    counts["weight"] = counts["weight"].clip(upper=max_weight)
    mean_w = (counts["weight"] * counts["n"]).sum() / counts["n"].sum()
    counts["weight"] /= mean_w
    return counts.set_index([tissue_col, sex_col])["weight"]


def apply_weights(rows, weight_table, sex_col="final_sex", tissue_col="OncotreeLineage",
                  default=1.0):
    #Look up the weight for each row. Groups not seen in training get `default`.
    idx = pd.MultiIndex.from_arrays([rows[tissue_col], rows[sex_col]])
    w = weight_table.reindex(idx).to_numpy(dtype=float)
    return pd.Series(np.where(np.isnan(w), default, w), index=rows.index, name="sample_weight")


if __name__ == "__main__":
    # Quick self-test: tissue A = 8 F + 2 M, tissue B = 5 M only
    toy = pd.DataFrame({
        "OncotreeLineage": ["A"] * 10 + ["B"] * 5,
        "final_sex": ["Female"] * 8 + ["Male"] * 2 + ["Male"] * 5,
    })
    table = fit_sex_tissue_weights(toy)
    print(table)
    w = apply_weights(toy, table)
    tot = toy.assign(w=w).groupby(["OncotreeLineage", "final_sex"])["w"].sum()
    print("\nTotal weight per group (A's two sexes should match):\n", tot)
    assert np.isclose(tot[("A", "Female")], tot[("A", "Male")])
    assert np.isclose(w.mean(), 1.0)
    print("\nSelf-test passed")