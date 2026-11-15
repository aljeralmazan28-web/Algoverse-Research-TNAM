# GDSC2 frozen split package — organization and independent review

This directory reorganizes the user-uploaded `final_gdsc2_splits (1).zip` into the agreed deliverable structure **without changing its original split assignments**. The two separately uploaded 526-cell-line CSVs are preserved in `reference_526/` and **not merged** with the ZIP outputs because their assignments differ.

## Main deliverables

```text
splits/
  unseen_cell_lines/
    train.csv
    val.csv
    test.csv
  unseen_drugs/
    train.csv
    val.csv
    test.csv
  unseen_tissues/
    train.csv
    val.csv
    test.csv
split_config.json
split_validation_report.csv
```

Each train/val/test file has `sample_id`, `cell_line_id`, `drug_id`, `tissue`, `sex`, `response`, and `split`. `sample_id` identifies a cell-line/drug measurement; don't confuse it with a cell-line-only ID. These files are observation-level partitions of the uploaded GDSC2 eligible cohort.

## Summary (verified from uploaded data)

- **125,360 response observations**, **492 distinct cell lines**, **295 drugs**, **22 tissues**; **GDSC2 only**.
- Unseen cell lines: 87,905 training / 18,833 validation / 18,622 testing observations; 344 / 74 / 74 held-out cell-line groups.
- Unseen drugs: 87,702 / 18,851 / 18,807 observations; 207 / 44 / 44 held-out drugs.
- Unseen tissues: 87,025 / 21,543 / 16,792 observations; 16 / 3 / 3 held-out tissues.
- The source ZIP's split-file and summary-file checksums match its manifest. No overlap of held-out group identifiers across train/val/test was detected, and no duplicate observation IDs were detected.

## Review items before declaring the experiment signed off

1. The two separately uploaded 526-cell-line CSVs contain **34 cell lines not included** in the ZIP's 492-line eligible cohort. The source Cellosaurus mapping audit indicates 20 ambiguous/unknown entries, 12 unique/unknown entries and 2 additional exclusions with non-unknown reported sex. Reasons for those last two need verification rather than guessing.
2. For the overlapping 492 cell lines, the standalone **cell-line** split disagrees with the ZIP for **225 cell lines**. The standalone **tissue** split disagrees with the ZIP for **271 cell lines**. A team decision is needed about which version Aljer's existing model checkpoints and baseline predictions used. The ZIP's assignments have been preserved here, not regenerated.
3. The ZIP's `sex` values agree with **reported_donor_sex** in its Cellosaurus mapping audit. It does not establish that Natasha's final *corrected/inferred sex* labels were applied.
4. The output contains GDSC2 only. It does not prove GDSC1/2 combined coverage.
5. Group exclusivity is proven, and sex/tissue proportions are reported. The original code that performed stratification/optimization was not supplied, so the procedure itself cannot be fully certified. No tissue-stratification across identical tissue labels is possible for unseen-tissue holdout.
6. The source manifest contains hashes for the original crosswalk, Cellosaurus TXT, and GDSC2 XLSX, but those raw inputs and split-generation script are not inside the ZIP. The configuration is faithful to the supplied manifest, not an independent demonstration that the original split generation can be rerun from scratch.
7. These are split assignments, **not** model predictions or an H2 evaluation harness. RMSE, Pearson/Spearman, gap CI, p-values and baseline performance-gap table require appropriately aligned predictions.

## Additional files

- `split_partition_summary.csv`: per-split row, group, sex, and tissue totals.
- `sex_tissue_distribution.csv`: stratification audit for each tissue in each partition.
- `source_manifest.json`: exact original source manifest.
- `source_data/`: exact original ZIP members, including eligible response cohort and mapping audits.
- `reference_526/`: the user's standalone Cellosaurus split files, unchanged.
- `comparisons/`: 34 excluded cell-line details and the differing assignments relative to the ZIP.

## Freeze workflow

Treat the ZIP assignment CSVs as a **candidate frozen GDSC2 version**, not as the team's signed-off canonical partitions, until mentor/team confirms which set Aljer used and the sex-label version. For future runs, distribute this packaged ZIP and keep it immutable. Check file checksums in `split_checksums.sha256` to detect changes.
