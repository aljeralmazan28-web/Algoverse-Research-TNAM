#Shared paths and settings for the sex-audit pipeline.
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data"
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
for d in (RAW, PROC, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# Input files (rename here if your downloads differ)
CELLOSAURUS = RAW / "cellosaurus.txt"
DEPMAP_MODEL = RAW / "Model.csv"
GDSC_LINES = RAW / "Cell_Lines_Details.xlsx"
EXPRESSION = RAW / "OmicsExpressionTPMLogp1HumanAllGenes.csv"
COPY_NUMBER = RAW / "OmicsCNGene.csv"          # optional

# Marker genes (no pseudoautosomal genes)
XIST = "XIST"
Y_GENES = ["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "ZFY", "USP9Y"]

# Sex-specific lineages (check names against Model.csv OncotreeLineage)
FEMALE_TISSUE = ["Ovary/Fallopian Tube", "Uterus", "Cervix", "Vulva/Vagina"]
MALE_TISSUE = ["Prostate", "Testis"]
SEX_SPECIFIC = FEMALE_TISSUE + MALE_TISSUE + ["Breast"]

# Analysis settings
CONF_POSTERIOR = 0.95        # how sure the mixture model must be for a "high confidence" call
MIN_LINES_PER_TISSUE = 10    # smaller lineages are pooled into "Other"

# --- Final labels & sex balance ---
Y_LOSS_CN = 0.3          # Y copy number below this = Y loss (check a histogram of y_cn first)
MIN_PER_SEX = 5          # fewer lines than this of either sex = can't compare sexes within that tissue
IMBALANCE_FRAC = 0.30    # flag tissues where the minority sex is < 30% of sexed lines
MAX_WEIGHT = 10.0        # cap on reweighting weights

# --- Figure colors: PLACEHOLDERS, replace with Kiran's hex codes ---
COLORS = {
    "female":   "#D1495B",
    "male":     "#00798C",
    "unknown":  "#BBBBBB",
    "mislabel": "#EDAE49",
    "missing":  "#8E7DBE",
    "accent":   "#333333",
}
FIG_DPI = 300