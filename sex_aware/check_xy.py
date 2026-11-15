# check_xy.py
import pandas as pd
from sex_aware import select_xy_columns

path = "/Users/natashaprabhu/PycharmProjects/Algoverse-Research-TNAM/sex_labels/data/raw_data/OmicsExpressionTPMLogp1HumanAllGenes.csv"     # CHANGE
labels = pd.read_csv("/Users/natashaprabhu/PycharmProjects/Algoverse-Research-TNAM/sex_labels/data/sex_labels_final.csv")    # CHANGE

# 1. which X/Y genes exist?
header = pd.read_csv(path, nrows=0).columns.tolist()
gene_names = [c.split(" (")[0] for c in header]
idx, found, missing = select_xy_columns(gene_names)
print("found:", found)
print("missing:", missing)

# 2. load just those columns, one row per cell line
xy_cols = [header[i] for i in idx]
df = pd.read_csv(path, usecols=["ModelID", "IsDefaultEntryForModel"] + xy_cols)
print("rows:", len(df), "| unique ModelID:", df["ModelID"].nunique())
df = df[df["IsDefaultEntryForModel"].astype(str).str.lower().isin(["yes", "true"])]
print("after default-entry filter:", len(df), "| unique:", df["ModelID"].nunique())

# 3. overlap with your labels
m = df.merge(labels[["ModelID", "final_sex"]], on="ModelID", how="inner")
print("labeled lines with expression:", len(m), "of", len(labels))

# 4. sanity check: XIST should be higher in females, Y genes higher in males
m.columns = [c.split(" (")[0] for c in m.columns]
print(m.groupby("final_sex")[found].mean().round(2))