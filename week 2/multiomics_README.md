
# Week 2 Multi-Omics Processing

## Data
Four aligned omics matrices:
- Expression: 526 x 18,025
- Mutation: 526 x 1,667
- Copy number: 526 x 23,264
- Proteomics: 526 x 8,498

All matrices use SIDM cell-line identifiers
and share identical sample ordering.

## Sex Metadata
492 of 526 cell lines have provisionally usable
sex labels based on name matching.

## Preprocessing
The reusable pipeline supports:
- Training-only feature selection
- Missing-value imputation
- Constant-feature removal
- Standardization of continuous features
- Consistent validation/test transformations

Proteomics uses a minimum 70% training
coverage threshold.

The pipeline was smoke-tested on a temporary
30-sample training and 10-sample evaluation subset.

Final model-ready outputs require the team's
approved train/validation/test splits.

## X/Y Gene Module
XIST, RPS4Y1, DDX3Y, KDM5D, UTY

All five candidates are present in the
aligned expression matrix.

## Exploratory Pathway Enrichment
Reactome preranked GSEA was conducted using
tissue-adjusted proteomics rankings.

High-confidence subset:
155 samples, 5 tissues, 2 significant pathways.

Same-tissue expanded subset:
278 samples, 5 tissues, 6 significant pathways.

Broader subset:
344 samples, 9 tissues, 29 significant pathways.

No pathway was significant across all three
analyses at FDR < 0.05.

PI3K/AKT-related pathways were significant only
in the high-confidence analysis.

Peptide hormone metabolism was significant
only in the broader analysis.

These are exploratory associations, not
validated sex-specific drug-response findings.

## Important Limitations
- Cell-line identity matching remains provisional.
- Some sex labels have low confidence.
- Proteomics includes missing measurements.
- Protein-to-gene mapping requires validation.
- The analyses share overlapping cell lines.
- GSEA results depend on sample selection.
- The omics matrices are aligned inputs, not
  final split-specific preprocessed outputs.
