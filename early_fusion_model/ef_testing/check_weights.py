import numpy as np
import pandas as pd
from reweighting import fit_sex_tissue_weights, apply_weights
from sex_aware import encode_sex

labels = pd.read_csv("/sex_labels/data/sex_labels_final.csv")

table = fit_sex_tissue_weights(labels)
w = apply_weights(labels, table)
print("one weight per line:", len(w) == len(labels))
print("mean: %.3f | min: %.3f | max: %.3f" % (w.mean(), w.min(), w.max()))
print("any NaN or zero:", w.isna().any() or (w == 0).any())

df = labels.assign(w=w)
print(df.groupby(["OncotreeLineage", "final_sex"]).agg(n=("w", "size"), weight=("w", "mean"), total=("w", "sum")).round(3))

feats, keep = encode_sex(labels["final_sex"].values)
print("female / male / unknown:", feats.sum(dim=0).tolist())