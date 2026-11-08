#Join DepMap, GDSC and Cellosaurus; harmonize reported sex labels.
import re
import numpy as np
import pandas as pd
from config import DEPMAP_MODEL, GDSC_LINES, PROC


def load_depmap():
    model = pd.read_csv(DEPMAP_MODEL)
    cols = ["ModelID", "CellLineName", "RRID", "SangerModelID", "COSMICID",
            "Sex", "OncotreeLineage", "OncotreePrimaryDisease"]
    missing = [c for c in cols if c not in model.columns]
    if missing:
        raise KeyError(f"Model.csv is missing columns {missing}; check your release")
    model = model[cols].copy()
    model["COSMICID"] = pd.to_numeric(model["COSMICID"], errors="coerce").astype("Int64").astype("string")
    return model


def load_gdsc():
    gdsc = pd.read_excel(GDSC_LINES, sheet_name=0)
    gdsc.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in gdsc.columns]
    print("GDSC columns:", gdsc.columns.tolist())
    gdsc = gdsc.rename(columns={"Sample Name": "gdsc_name", "COSMIC identifier": "cosmic_id"})
    gdsc["cosmic_id"] = pd.to_numeric(gdsc["cosmic_id"], errors="coerce")
    gdsc = gdsc.dropna(subset=["cosmic_id"])                 # drops TOTAL row
    gdsc["cosmic_id"] = gdsc["cosmic_id"].astype(int).astype(str)
    if "Drug Response" in gdsc.columns:                      # screened lines only
        gdsc = gdsc[gdsc["Drug Response"] == "Y"]
    return gdsc


def norm_sex(s):
    if pd.isna(s):
        return "Unknown"
    s = str(s).strip().lower()
    return {"female": "Female", "male": "Male", "mixed sex": "Mixed"}.get(s, "Unknown")


def metadata_status(r):
    labels = [r["sex_depmap"], r["sex_cello"]]
    known = [s for s in labels if s != "Unknown"]
    if not known:
        return "missing"
    if len(set(known)) > 1:
        return "conflict"
    return "agree" if len(known) == 2 else "single-source"


def reported_sex(r):
    if r["metadata_status"] in ("agree", "single-source"):
        return r["sex_depmap"] if r["sex_depmap"] != "Unknown" else r["sex_cello"]
    return "Conflict" if r["metadata_status"] == "conflict" else "Unknown"


def main():
    model = load_depmap()
    gdsc = load_gdsc()
    cello = pd.read_csv(PROC / "cellosaurus_human.csv", dtype=str)

    # Map DepMap -> Cellosaurus: RRID first, Cellosaurus cross-reference as fallback
    c_dep = (cello.assign(depmap_id=cello["depmap_id"].str.split(";"))
                  .explode("depmap_id").dropna(subset=["depmap_id"]))
    m = model.merge(c_dep[["cvcl", "depmap_id"]], left_on="ModelID",
                    right_on="depmap_id", how="left")
    dup = m["ModelID"].duplicated(keep=False)
    if dup.any():
        print("WARNING: DepMap IDs mapping to >1 Cellosaurus entry:")
        print(m.loc[dup, ["ModelID", "CellLineName", "cvcl"]])
        m = m.drop_duplicates("ModelID")

    m["cvcl_final"] = m["RRID"].fillna(m["cvcl"])
    m["map_method"] = np.select([m["RRID"].notna(), m["cvcl"].notna()],
                                ["RRID", "cellosaurus_xref"], "unmapped")
    m["id_conflict"] = m["RRID"].notna() & m["cvcl"].notna() & (m["RRID"] != m["cvcl"])

    m = m.drop(columns=["cvcl", "depmap_id"]).merge(
        cello[["cvcl", "sex_cellosaurus", "amelogenin", "problematic", "cosmic_clp"]],
        left_on="cvcl_final", right_on="cvcl", how="left").drop(columns="cvcl")

    m["in_gdsc"] = m["COSMICID"].isin(set(gdsc["cosmic_id"]))

    # Harmonize sex labels
    m["sex_depmap"] = m["Sex"].map(norm_sex)
    m["sex_cello"] = m["sex_cellosaurus"].map(norm_sex)
    m["metadata_status"] = m.apply(metadata_status, axis=1)
    m["reported_sex"] = m.apply(reported_sex, axis=1)

    m.to_csv(PROC / "reported_sex_main.csv", index=False)

    # Checks
    print(m["map_method"].value_counts())
    print("ID conflicts (RRID vs Cellosaurus):", m["id_conflict"].sum())
    both = m.dropna(subset=["COSMICID", "cosmic_clp"])
    print("COSMIC ID mismatches:", (both["COSMICID"] != both["cosmic_clp"]).sum())

    missing = gdsc[~gdsc["cosmic_id"].isin(set(m["COSMICID"].dropna()))]
    missing.to_csv(PROC / "gdsc_not_in_depmap.csv", index=False)
    print(f"GDSC lines: {len(gdsc)}, in DepMap: {m['in_gdsc'].sum()}, not in DepMap: {len(missing)}")

    print("\nDepMap vs Cellosaurus sex:")
    print(pd.crosstab(m["sex_depmap"], m["sex_cello"]))
    print("\nMetadata status (GDSC lines):")
    print(m.loc[m["in_gdsc"], "metadata_status"].value_counts())


if __name__ == "__main__":
    main()