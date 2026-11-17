"""
make_synthetic_inputs.py -- build a small FAKE input folder in the same layout as the real one,
so run_early_fusion.py can be tested end to end before the real files arrive.

    python make_synthetic_inputs.py --out ~/ef_fake_inputs
    python run_early_fusion.py --input-dir ~/ef_fake_inputs --out-dir out_fake \
        --splits unseen_cell_lines unseen_tissues --variants base a b c abc --seeds 1 2 --max-epochs 3
    python run_early_fusion.py --input-dir ~/ef_fake_inputs --out-dir out_fake \
        --variants a b abc --seeds 1 2 --max-epochs 3 --control both

The numbers it produces mean NOTHING (random data). It only checks that every switch, the seeds,
the controls and the output files work. The fake X/Y genes do carry sex, like the real ones.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MARKERS = ["XIST", "RPS4Y1", "DDX3Y", "KDM5D", "UTY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-cells", type=int, default=150)
    ap.add_argument("--n-drugs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    n = a.n_cells
    cells = [f"SIDM{i:05d}" for i in range(n)]
    tissue_names = ["Lung", "Skin", "Brain", "Bowel", "Kidney", "Liver"]
    tissue = np.array([tissue_names[i % 6] for i in range(n)])
    sex = np.where(rng.random(n) < 0.45, "Female", "Male").astype(object)
    sex[:3] = "Unknown"                                   # a few unknown-sex lines, dropped by QC
    female = sex == "Female"

    def table(cols, binary=False, nan=0.0):
        arr = (rng.random((n, len(cols))) < 0.1).astype(np.float32) if binary \
            else rng.normal(size=(n, len(cols))).astype(np.float32)
        df = pd.DataFrame(arr, index=cells, columns=cols)
        if nan:
            df = df.mask(rng.random(df.shape) < nan)
        return df

    expr = table([f"g{i}" for i in range(60)] + MARKERS)
    expr.loc[female, "XIST"] += 4.0
    expr.loc[sex == "Male", ["RPS4Y1", "DDX3Y", "KDM5D", "UTY"]] += 4.0
    mut = table([f"m{i}" for i in range(40)], binary=True)
    cnv = table([f"c{i}" for i in range(40)])
    prot = table([f"p{i}" for i in range(30)], nan=0.05)
    for name, df in [("expression", expr), ("mutation", mut), ("cnv", cnv), ("proteomics", prot)]:
        df.rename_axis("SIDM").reset_index().to_csv(out / f"{name}_aligned.csv", index=False)

    pd.DataFrame({"SIDM": cells, "final_sex": sex, "usable_sex": sex != "Unknown",
                  "exclude_from_sex_analysis": False, "label_confidence": "high"}
                 ).to_csv(out / "sex_labels_matched.csv", index=False)

    drugs = [f"{1000 + i}" for i in range(a.n_drugs)]
    drug_eff = rng.normal(size=a.n_drugs)
    recs = []
    for i, c in enumerate(cells):
        for j, d in enumerate(drugs):
            y = 0.8 * expr.iloc[i, 0] - 0.5 * expr.iloc[i, 1] + drug_eff[j] \
                + (0.5 if female[i] else 0.0) + rng.normal(scale=0.5)
            recs.append(dict(sample_id=f"{c}__{d}", cell_line_id=c, drug_id=d, tissue=tissue[i],
                             sex=sex[i], response=float(y)))
    allrows = pd.DataFrame(recs)

    # unseen cell lines: random 70/15/15 by cell line
    order = rng.permutation(n)
    groups = {"train": order[:int(.7 * n)], "val": order[int(.7 * n):int(.85 * n)], "test": order[int(.85 * n):]}
    # unseen tissues: 4 tissues train, 1 val, 1 test
    tgroups = {"train": tissue_names[:4], "val": tissue_names[4:5], "test": tissue_names[5:]}
    for family, parts in [("unseen_cell_lines", {p: allrows[allrows.cell_line_id.isin([cells[i] for i in idx])]
                                                  for p, idx in groups.items()}),
                          ("unseen_tissues", {p: allrows[allrows.tissue.isin(ts)] for p, ts in tgroups.items()})]:
        d = out / "splits" / family
        d.mkdir(parents=True, exist_ok=True)
        for p, df in parts.items():
            df.assign(split=p).to_csv(d / f"{p}.csv", index=False)
    print(f"Wrote fake inputs to {out}  ({n} cell lines, {a.n_drugs} drugs). Results on this data mean nothing.")


if __name__ == "__main__":
    main()