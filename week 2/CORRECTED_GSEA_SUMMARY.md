# Week 2: Corrected Proteomics Pathway Enrichment

## Dataset
526 cell lines aligned across four omics layers.
492 cell lines have provisionally usable sex labels.

## Protein-to-Gene Mapping
- 8,498 original protein identifiers
- 8,467 human protein entries
- 8,458 mapped to UniProt primary gene names
- 9 human entries without mapped gene names
- One duplicate gene (MACF1), resolved by selecting
  the protein with the largest absolute t-statistic

UniProt primary gene names were used instead of
protein entry-name abbreviations.

## Reactome GSEA
Method: Tissue-adjusted differential-protein rankings,
followed by preranked Reactome GSEA.

Settings:
- Minimum gene-set size: 15
- Maximum gene-set size: 500
- Permutations: 1,000
- Seed: 42
- Ranking statistic: signed t-statistic

## Corrected Results
- High-confidence (155): 39 significant pathways
- Same tissues (278): 66 significant pathways
- Broader (344): 96 significant pathways

- Significant across all three: 24
- Consistent enrichment direction: 24

## Key Findings
- GPER1 signaling was significant with positive
  enrichment in all three analyses.
- Negative regulation of the PI3K/AKT network was
  significant with negative enrichment in all three.
- Mitochondrial translation pathways were positively
  enriched in all three.
- Peptide hormone metabolism was significant only
  in the broader 344-cell-line analysis.

## Limitations
- Analyses share cell lines and are not independent
  biological replications.
- Sex labels include provisional, lower-confidence matches.
- Tissue adjustment does not eliminate all confounding.
- Protein-to-gene mapping uses UniProt primary gene names,
  which are not necessarily current HGNC-approved symbols.
- Multiple Reactome pathways share genes.
- Zero-valued reported permutation FDR estimates
  should not be interpreted as zero uncertainty.
- Enrichment does not prove pathway activity,
  causality, or differences in drug response.
- These are exploratory results.

## Data Processing
Aligned matrices and the X/Y candidate module are
provided separately.

Training-specific imputation, feature selection and
scaling must be fitted on training samples only.
