#Small helpers shared across scripts
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


def pool_lineages(lineage, min_n):
    #Pool lineages with fewer than min_n lines into 'Other'.
    lineage = lineage.fillna("Unknown lineage")
    size = lineage.map(lineage.value_counts())
    return lineage.where(size >= min_n, "Other")


def wilson(k, n):
    #95% Wilson CI as percentages.
    if n == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(k, n, method="wilson")
    return 100 * lo, 100 * hi