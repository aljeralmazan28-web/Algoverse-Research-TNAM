"""
rescore_labels.py -- rescore saved predictions on a different sex-label file, no retraining.

The runner scores each model on `final_sex` (all usable lines). The team harness can use a stricter file
(`corrected_sex_labels.csv`: only confidently labeled lines). This script recomputes overall / female / male
RMSE and the gap from the prediction files for ANY label file, and writes a runs.csv-style table that
analyze_h4.py can read.

    python rescore_labels.py out_main/predictions out_control/predictions \
        --labels <repo>/harness_results/corrected_sex_labels.csv --out runs_strict.csv
    python analyze_h4.py runs_strict.csv --split unseen_cell_lines --out h4_table_strict.csv

Prediction files must be named like the runner writes them:
    EF_<variant>[_prot][_shuf<sex|xy|both>]_s<seed>__<split>.csv   with columns cell_line_id, y_pred, y_true.
Label file needs cell_line_id and one of corrected_sex / final_sex / sex. Lines not Female or Male are dropped.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

NAME = re.compile(r"^EF_(base|a|b|c|abc)(_prot)?(?:_shuf(sex|xy|both))?_s(\d+)__(.+)\.csv$")


def rmse(d):
    return float(np.sqrt(((d["y_pred"] - d["y_true"]) ** 2).mean()))


def load_labels(path):
    lab = pd.read_csv(path)
    col = next((c for c in ("corrected_sex", "final_sex", "sex") if c in lab.columns), None)
    if col is None or "cell_line_id" not in lab.columns:
        raise SystemExit("Label file needs cell_line_id and one of corrected_sex / final_sex / sex.")
    lab = lab[["cell_line_id", col]].rename(columns={col: "sex_label"})
    if lab["cell_line_id"].duplicated().any():
        raise SystemExit("Label file has duplicate cell_line_id.")
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dirs", nargs="+", type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs_rescored.csv"))
    ap.add_argument("--split", default=None, help="only this split, e.g. unseen_cell_lines")
    a = ap.parse_args()

    lab = load_labels(a.labels)
    rows, skipped = [], []
    for d in a.pred_dirs:
        for f in sorted(d.expanduser().glob("*.csv")):
            m = NAME.match(f.name)
            if not m:
                skipped.append(f.name)
                continue
            variant, prot, ctl, seed, split = m.groups()
            if a.split and split != a.split:
                continue
            p = pd.read_csv(f).merge(lab, on="cell_line_id", how="left")
            p = p[p["sex_label"].isin(["Female", "Male"])]
            fm, mm = p[p.sex_label == "Female"], p[p.sex_label == "Male"]
            rows.append(dict(
                model=f.name.split("__")[0], split_family=split, variant=variant, seed=int(seed),
                proteomics=bool(prot), control=ctl or "none", overall_rmse=rmse(p),
                female_rmse=rmse(fm), male_rmse=rmse(mm),
                gap_female_minus_male=rmse(fm) - rmse(mm), gap_abs=abs(rmse(fm) - rmse(mm)),
                n_test=len(p), n_lines=p["cell_line_id"].nunique(), female_n=len(fm), male_n=len(mm)))
    if not rows:
        raise SystemExit("No prediction files matched the expected names.")
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(f"Rescored {len(out)} prediction files on {a.labels.name}: "
          f"{int(out['n_test'].median()):,} rows and {int(out['n_lines'].median())} cell lines per file (median).")
    if skipped:
        print(f"Skipped {len(skipped)} files with unexpected names, e.g. {skipped[0]}")
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()