import pandas as pd

def read_cellosaurus(path):
    #Return a list of entries; each entry is {line_code: [values]}
    entries, entry = [], {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "//":                       # end of an entry
                if "AC" in entry:
                    entries.append(entry)
                entry = {}
            elif len(line) > 5 and line[2:5] == "   ":   # XX   value
                entry.setdefault(line[:2], []).append(line[5:])
    return entries

entries = read_cellosaurus("data/raw/cellosaurus.txt")
print(len(entries))

def first(e, code):
    return e.get(code, [None])[0]

def xref(e, db):
    # DR lines look like "DepMap; ACH-000940"
    ids = [v.split("; ", 1)[1] for v in e.get("DR", []) if v.startswith(db + "; ")]
    return ";".join(ids) if ids else None

def amelogenin(e):
    vals = [v.split("Amelogenin:", 1)[1].strip() for v in e.get("ST", []) if "Amelogenin" in v]
    if not vals:
        return None
    has_y = ["Y" in v for v in vals]
    if all(has_y):
        return "X,Y"
    if not any(has_y):
        return "X"
    return "conflicting"          # different sources disagree

def problematic(e):
    for v in e.get("CC", []):
        if v.startswith("Problematic cell line:"):
            return v.split(":", 1)[1].strip()
    return None

def parent(e):
    hi = first(e, "HI")           # "CVCL_0034 ! NTERA-2"
    return hi.split(" ! ")[0] if hi else None

rows = []
for e in entries:
    if "NCBI_TaxID=9606" not in (first(e, "OX") or ""):   # human only
        continue
    rows.append({
        "cvcl": first(e, "AC"),
        "name": first(e, "ID"),
        "synonyms": first(e, "SY"),
        "category": first(e, "CA"),
        "sex_cellosaurus": first(e, "SX"),
        "depmap_id": xref(e, "DepMap"),
        "cosmic_clp": xref(e, "Cosmic-CLP"),
        "sanger_id": xref(e, "Cell_Model_Passport"),
        "amelogenin": amelogenin(e),
        "problematic": problematic(e),
        "parent": parent(e),
    })

cello = pd.DataFrame(rows)
cello.to_csv("data/processed/cellosaurus_human.csv", index=False)

print(cello.shape)
print(cello["sex_cellosaurus"].value_counts(dropna=False))
print(cello[["depmap_id", "cosmic_clp", "sanger_id"]].notna().sum())

# Known lines: HeLa and MCF-7 should be Female / X; A549 should be Male / X,Y
print(cello[cello["name"].isin(["HeLa", "MCF-7", "A549"])])

# Lines with more than one DepMap ID need a manual look
print(cello[cello["depmap_id"].str.contains(";", na=False)])
