# Week 2: Multi-Omics Preprocessing and Pathway Enrichment

## Project Overview

This work supports our research project, **Using Sex-Stratification and Multi-Omics Data to Predict Drug Response**. The goal of Week 2 was to prepare aligned molecular datasets, develop a reproducible preprocessing pipeline, analyze sex-associated differences in protein measurements, and identify potentially relevant biological pathways.

The prepared datasets and preprocessing code will be used for subsequent drug-response prediction models.

## 1. Dataset Preparation and Alignment

Four omics datasets were aligned using shared cell-line identifiers.

| Omics Layer | Cell Lines | Features |
|---|---:|---:|
| Gene expression | 526 | 18,025 |
| Mutation | 526 | 1,667 |
| Copy number variation (CNV) | 526 | 23,264 |
| Proteomics | 526 | 8,498 |

All four matrices use SIDM identifiers and contain the same 526 cell lines in identical order.

The overlapping cell lines were identified using the team's provisional GDSC and multi-omics mapping.

**Sex metadata:**
- 526 cell lines included in the aligned dataset.
- 492 cell lines have provisionally usable sex labels.
- Sex labels were matched using normalized cell-line names.
- Some sex labels have lower confidence and require cautious interpretation.

The 526 aligned samples are not necessarily confirmed to have usable drug-response measurements for every drug.

## 2. Multi-Omics Preprocessing

A reusable Python preprocessing pipeline was developed for the four omics layers.

### Preprocessing Methods

**Gene Expression**
- Remove features with no usable training measurements.
- Median-impute missing measurements.
- Remove constant features.
- Apply standardization using training-set statistics.

**Mutation**
- Remove features with no usable training measurements.
- Median-impute missing measurements.
- Remove constant features.
- Preserve unscaled mutation values.

**Copy Number Variation**
- Remove features with no usable training measurements.
- Median-impute missing measurements.
- Remove constant features.
- Preserve unscaled copy-number values.

**Proteomics**
- Retain proteins with at least 70% coverage in the training set.
- Apply median imputation for missing measurements.
- Remove constant features.
- Standardize using training-set statistics.

The Harmonizome expression data contains processed association scores rather than raw RNA expression measurements. Mutation and CNV data also use encoded values.

### Validation

The preprocessing pipeline was tested using a temporary subset of 30 training samples and 10 evaluation samples.

All four omics layers passed the following checks:
- Training and evaluation samples contained identical selected features.
- All transformed values were finite.
- No missing values remained after transformation.
- Standardized training features had approximately zero mean.
- Original aligned datasets retained all 526 samples.

These checks demonstrate that the pipeline executes correctly on the test subset. They do not replace validation using the final model-training splits.

**Important:** Preprocessing parameters must be fitted on training samples only and subsequently applied to validation and test samples to prevent data leakage. The exported aligned matrices are intermediate inputs, not final split-specific model-ready matrices.

## 3. Proteomics Processing and Gene Mapping

The proteomics dataset originally contained 8,498 protein identifiers.

Protein identifiers were converted to gene symbols using UniProt accession numbers instead of directly extracting abbreviated protein entry names.

### Mapping Results

| Mapping Category | Count |
|---|---:|
| Total protein identifiers | 8,498 |
| Human protein entries | 8,467 |
| Human entries successfully mapped | 8,458 |
| Human entries without mapped gene symbols | 9 |
| Unique mapped gene symbols | 8,457 |
| Unique mapped symbols found in Reactome | 5,886 |

The corrected mapping increased Reactome gene-symbol matching from **33.96% to 69.60%**.

One duplicated gene symbol, MACF1, corresponded to two protein identifiers. For GSEA rankings, the protein measurement with the largest absolute t-statistic was retained.

The 31 non-human protein entries were excluded from the human gene mapping.

UniProt primary gene names were used for this analysis; these are not necessarily identical to current HGNC-approved gene symbols.

## 4. Sex-Associated Differential Proteomics

Exploratory differential proteomics analyses were performed using tissue-adjusted statistical models.

Three sample selections were evaluated:

| Analysis | Cell Lines | Tissue Groups |
|---|---:|---:|
| High-confidence sex labels | 155 | 5 |
| Expanded same-tissue subset | 278 | 5 |
| Broader tissue-balanced subset | 344 | 9 |

Protein-level statistics were used to rank genes for Reactome Gene Set Enrichment Analysis (GSEA).

The ranking statistic was the signed tissue-adjusted t-statistic, where positive values indicate higher protein measurements in female-labeled cell lines relative to male-labeled cell lines.

The analyses are exploratory and do not establish causality or drug-response differences.

## 5. Corrected Reactome Pathway Enrichment

Reactome preranked GSEA was performed using corrected UniProt gene mappings and tissue-adjusted differential protein statistics.

### GSEA Parameters

- Gene-set database: Reactome
- Minimum gene-set size: 15
- Maximum gene-set size: 500
- Permutations: 1,000
- Random seed: 42
- Ranking statistic: Signed t-statistic
- Significance threshold: FDR < 0.05

### Corrected Results

| Analysis | Pathways Tested | Significant Pathways |
|---|---:|---:|
| High-confidence (155 samples) | 967 | 39 |
| Same tissues (278 samples) | 997 | 66 |
| Broad (344 samples) | 974 | 96 |

**Twenty-four pathways were significant across all three analyses, and all 24 showed consistent enrichment directions.**

### Notable Pathway Findings

**GPER1 Signaling**

GPER1 signaling demonstrated significant positive enrichment across all three sample groups.

- High-confidence: NES = +2.301
- Same tissues: NES = +2.305
- Broad: NES = +2.234

GPER1 is associated with estrogen-responsive signaling, making it relevant to the project's investigation of sex-associated molecular patterns. However, enrichment alone does not establish hormone activity or causality.

**Negative Regulation of the PI3K/AKT Network**

This pathway demonstrated significant negative enrichment across all three analyses.

- High-confidence: NES = -2.310
- Same tissues: NES = -2.072
- Broad: NES = -1.804

The pathway remained significant across sample selections, although the enrichment magnitude decreased in the broader analysis.

**Mitochondrial Translation**

Mitochondrial translation demonstrated significant positive enrichment across all three analyses.

- High-confidence: NES = +2.160
- Same tissues: NES = +3.039
- Broad: NES = +2.964

Related mitochondrial translation initiation, elongation, and termination pathways also showed consistent enrichment.

**Peptide Hormone Metabolism**

Peptide hormone metabolism was significant only in the broader 344-cell-line analysis.

- High-confidence: NES = +1.071, FDR = 0.7772
- Same tissues: NES = +1.293, FDR = 0.2606
- Broad: NES = +1.892, FDR = 0.0066

This finding is sensitive to sample selection and should not be described as consistently enriched.

### Interpretation

The corrected results suggest that GPER1 signaling, PI3K/AKT regulation, and mitochondrial pathways may be relevant to sex-associated proteomic differences in the analyzed cancer cell lines.

Consistency across three overlapping sample selections supports further investigation but does not constitute independent replication.

The pathway results do not demonstrate that these molecular differences cause variation in drug response.

## 6. X/Y Gene Module

Five candidate sex-associated genes were identified in the aligned gene-expression dataset:

- XIST
- RPS4Y1
- DDX3Y
- KDM5D
- UTY

All five genes were confirmed to exist in the aligned expression matrix.

The candidate list was exported as `XY_gene_module_candidates.csv`.

These genes are potential features for sex-aware modeling. Their inclusion and interpretation should account for possible overlap with the markers used to infer sex labels.

## 7. Week 2 Deliverables

### Code

**`multiomics_preprocessing.ipynb`**

Notebook containing the multi-omics alignment, exploratory statistical analysis, pathway enrichment, identifier mapping, and validation steps.

**`multiomics_preprocessing_pipeline.py`**

Reusable preprocessing pipeline supporting training-only feature selection, imputation, constant-feature removal, and scaling.

### Aligned Dataset Package

**`week2_aligned_omics_package.zip`**

Contains:
- `expression_aligned.csv`
- `mutation_aligned.csv`
- `cnv_aligned.csv`
- `proteomics_aligned.csv`
- `sample_metadata.csv`
- `sex_labels_matched.csv`
- `XY_gene_module_candidates.csv`
- `README.txt`

### Corrected Pathway Results

**`week2_corrected_GSEA_results.zip`**

Contains the corrected gene rankings, Reactome GSEA results, significant pathway tables, shared pathway comparison, and UniProt protein-to-gene mapping.

**`CORRECTED_GSEA_SUMMARY.md`**

Provides a shorter summary of the corrected enrichment analysis and its limitations.

The dataset ZIP files should be stored in shared research storage, with access provided to the team. They do not need to be committed directly to GitHub.

## 8. Limitations and Future Work

- Cell-line and sex-label matching remains provisional.
- Some sex labels have lower confidence.
- Tissue adjustment may not eliminate all confounding.
- The three GSEA analyses contain overlapping samples and therefore do not represent independent replication.
- Multiple Reactome pathways share genes and may reflect overlapping biological signals.
- UniProt gene mapping improves identifier accuracy but does not guarantee current HGNC nomenclature.
- Protein-to-gene duplicates were resolved using a defined exploratory ranking rule.
- GSEA values displayed as zero do not imply zero statistical uncertainty.
- The harmonized molecular datasets contain processed and encoded measurements rather than uniformly raw omics measurements.
- The 526-cell-line overlap does not confirm drug-response availability for every cell-line–drug pair.
- Final imputation, scaling, and feature selection must use training-only data from the team's approved modeling splits.
- Pathway enrichment identifies statistical associations, not causal mechanisms or demonstrated differences in drug sensitivity.

## 9. Week 2 Completion Summary

Week 2 produced four aligned omics datasets covering 526 cancer cell lines, a reusable and tested preprocessing pipeline, a corrected UniProt protein-to-gene mapping, exploratory differential proteomics analyses, Reactome pathway enrichment results, and an exported X/Y candidate gene module.

The corrected GSEA identified 24 pathways significant across all three sample selections with consistent enrichment directions.

These outputs provide the data preparation and exploratory biological analysis needed for subsequent sex-aware, multi-omics drug-response modeling. Final split-specific preprocessing and predictive validation remain part of the downstream modeling workflow.
