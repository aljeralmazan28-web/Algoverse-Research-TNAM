import argparse
import sys
import pandas as pd

# PROVISIONAL list (Week 1 sex-inference genes). Keep in sync with sex_aware.py.
PROVISIONAL_XY_GENES = ["XIST", "RPS4Y1", "DDX3Y", "KDM5D", "UTY"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expression", required=True, help="full DepMap-style expression CSV")
    ap.add_argument("--out", default="xy_expression.csv")
    ap.add_argument("--genes", default=None, help="text file, one gene symbol per line")
    ap.add_argument("--labels", default=None, help="sex_labels_final.csv, for a sanity check")
    ap.add_argument("--require-all", action="store_true", help="exit with an error if any gene is missing")
    a = ap.parse_args()

    genes = PROVISIONAL_XY_GENES
    if a.genes:
        with open(a.genes) as f:
            genes = [ln.strip() for ln in f if ln.strip()]
    print(f"gene list: {'file ' + a.genes if a.genes else 'PROVISIONAL'} ({len(genes)} genes)")

    # 1. find the columns ("XIST (7503)" -> "XIST")
    header = pd.read_csv(a.expression, nrows=0).columns.tolist()
    clean = [c.split(" (")[0] for c in header]
    pos = {}
    for i, g in enumerate(clean):
        pos.setdefault(g, i)                       # first match wins
    found = [g for g in genes if g in pos]
    missing = [g for g in genes if g not in pos]
    print("found:  ", found)
    print("missing:", missing)
    if not found or (missing and a.require_all):
        sys.exit("ERROR: gene(s) missing -- check naming (symbol vs ENSG ID) before using this table")
    cols = [header[pos[g]] for g in found]

    # 2. load only those columns; keep one row per cell line
    meta = ["ModelID"]
    if "IsDefaultEntryForModel" in header:
        meta.append("IsDefaultEntryForModel")
    df = pd.read_csv(a.expression, usecols=meta + cols)
    print(f"rows read: {len(df)} | unique ModelID: {df['ModelID'].nunique()}")
    if "IsDefaultEntryForModel" in df.columns:
        keep = df["IsDefaultEntryForModel"].astype(str).str.lower().isin(["yes", "true"])
        df = df[keep].drop(columns="IsDefaultEntryForModel")
    if df["ModelID"].duplicated().any():
        sys.exit("ERROR: duplicate ModelID rows remain after the default-entry filter")
    # read_csv(usecols=...) returns columns in FILE order, not the order requested, so
    # rename by NAME (never by position) and then put them in gene-list order.
    df = df.rename(columns=dict(zip(cols, found)))[["ModelID"] + found]
    df = df.set_index("ModelID").sort_index()
    print(f"rows after default-entry filter: {len(df)} (one per cell line)")
    print(f"NaN values: {int(df.isna().sum().sum())}")

    df.to_csv(a.out)
    print(f"saved {a.out}  shape={df.shape}")

    # 3. optional sanity check against the sex labels
    if a.labels:
        lab = pd.read_csv(a.labels)[["ModelID", "final_sex"]]
        m = df.reset_index().merge(lab, on="ModelID", how="inner")
        print(f"labeled lines with X/Y data: {len(m)} of {len(lab)}")
        print("mean expression by final_sex (XIST high in Female, Y genes high in Male):")
        mm = m.groupby("final_sex")[found].mean()
        print(mm.round(2))
        # automatic alignment check: catches mislabeled columns or misaligned rows
        if {"Female", "Male"} <= set(mm.index):
            if "XIST" in found and not mm.loc["Female", "XIST"] > mm.loc["Male", "XIST"]:
                sys.exit("ERROR: XIST is not higher in Female -- columns or labels are misaligned. Do NOT use this file.")
            for g in [x for x in found if x in ("RPS4Y1", "DDX3Y", "KDM5D", "UTY")]:
                if not mm.loc["Male", g] > mm.loc["Female", g]:
                    sys.exit(f"ERROR: {g} is not higher in Male -- columns or labels are misaligned. Do NOT use this file.")
            print("alignment check passed: XIST higher in Female, Y genes higher in Male")


if __name__ == "__main__":
    main()