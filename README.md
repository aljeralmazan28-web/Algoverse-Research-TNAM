# Algoverse-Research-TNAM
Repository for research 8/23 fall cohort

Using Sex-stratification and multi-omics data to predict drug response
Instructions
Proposals must be specific and detailed so the direction of the research is clear. If you’re not sure about a given item, please mark it as such and we can discuss.
Add text using your assigned color for all tabs in this doc:
mentor, natasha prabhu, aljer almazan, malavika venkatesh, trishna erukulla
Relevant Past Papers
How is it done today, and what are the limits of current practice?
Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells | Nucleic Acids Research | Oxford Academic + The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity - PubMed
Large cancer cell line panels that pair drug sensitivity with genomics/transcriptomics
Gap: Sex metadata is not well recorded, can be wrong, almost never used as a variable when modeling

MOLI: multi-omics late integration with deep neural networks for drug response prediction | Bioinformatics | Oxford Academic
Late-fusing deep model that encodes mutation, copy number, and expression separately before combining them. This approach made this model outperform single-omic models
Gap: only one fusion strategy, does not use proteomics, performance was not evaluated across demographics

DeepCDR: a hybrid graph convolutional network for predicting cancer drug response | Bioinformatics | Oxford Academic 
Graph convolutional network on drug structure + multi-omics cell data (mutation, methylation, etc)
Gap: fusion design is fixed not benchmarked, no fairness analysis 

Pan-cancer proteomic map of 949 human cell lines - PubMed 
Proteomic profiles of hundreds of cell lines, proteomics can give you a better signal compared to just using transcriptomics
Gap: proteomics has not been compared across fusion methods or sexes

These next two papers are not related to AI/ML but are good background for the project. We could apply the findings of these papers to our project

Comprehensive Characterization of Molecular Differences in Cancer between Male and Female Patients - PubMed 
TCGA analysis; finding sex-based molecular signatures across cancers

Sex differences in pharmacokinetics predict adverse drug reactions in women - PubMed
Women have extreme drug reactions 2x as often as men, which can be explained by sex differences
Motivation	
Current drug-response models are trained on cell line panels but do not account for the following
are the panels balanced by sex?
Are these labels correct?
Does the model perform differently on samples from males vs females?
Sex metadata in cell line repositories is incomplete and TCGA shows that there are sex based differences in expression in drug targets. Also, women experience more adverse drug reactions

This is important because if a model is systematically less accurate for samples from females, then the drug prioritization based on that model will be less reliable for women.

Solution: Difference omics layers (transcriptomics, genomics, etc) contain sex info in different ways. If we compare fusion methods and factor in sex-stratification then we might be able to see how different methods preserve and discard the signal. 
Key Ideas/Contributions/Novelty
We could compare reported sex with chromosomal sex (which would be inferred based on expression), then we could measure/quantify mislabeling by tissue type
We could test different fusion strategies and create a benchmark, which every result is reported separately based on sex
We could also test how adding sex info affects accuracy or the performance gap between sexes
We could see which omics layers/pathways are the ones that dictate sex-based predictions and provide a ranked list of drugs with a prediction on how those drugs would affect each sex
So in summary:
Benchmark (1,2)
Method improvement (3)
Biological findings (4)
Methods
Infer sex from XIST and Y gene expression and compare it to metadata → record any differences and how often they occur per tissue type
Preprocess datasets, pathway enrichment
Fusion benchmark - early, late, multimodal, graph model
For each fusion model: sex as input + X/Y gene module + sex-balanced reweighting 
Evaluate and interpret results :D
Experimental Setup
Note: used claude to assist with this
Baseline
Expression only model
Could use MOLI or DeepCDR (see lit review)
Fusion models without the sex-aware components
Train/test split
Test cell lines, drugs, and tissues that were not seen during training separately 
The splits can be stratified by sex and tissue
Ablation
Shuffle sex labels
Remove X/Y genes
Remove each omics layer one at a time to see how much each layer contributes to the sex gao
Compare female and male lines within the same tissue (so like compare female and male skin tissue, not female skin tissue and male brain tissue)
Datasets and Evaluation
Cellosaurus
CCLE for proteomics stuff 
GDSC1 and GDSC2 for drug info
Benchmarks/Evaluation Sets
Compare against MOLI and DeepCDR (see lit review)
The fours fusion model w/ and w/o sex-aware factors
Ideal Results
Hypotheses
Sex is mislabeled or missing for a large fraction of cell lines, and female-derived lines are underrepresented in non-sex-specific tissues.
Standard models show a quantifiable, statistically significant performance gap between sexes
Adding proteomics reduces the gap, because they retain modality-specific (including hormone-related) signals.
Sex-aware components reduce the gap without lowering overall accuracy.
A small set of drugs shows robust sex-differential response that is supported by pathway attributions and prior literature.
Potential Limitations
Note: I got these from claude
Biological: Cell lines grow without sex hormones, so they capture chromosomal sex effects but miss hormonal and pharmacokinetic effects, which are a major source of clinical sex differences. This limits clinical interpretation and should be stated clearly.
Data: The proteomics overlap is small, which reduces statistical power. Some tissue types may have too few lines of one sex for reliable within-tissue comparison. Sex is treated as binary chromosomal sex because that is what the data supports; gender is not measurable here.
Confounding: Sex correlates with tissue, cancer subtype, ancestry, and age at derivation. Our controls reduce this confounding but cannot remove it entirely.
Generalization: Cell line findings may not transfer to patients; external validation depends on access to patient data.
Compute: Transformers and GNNs across many splits, seeds, and ablations are expensive. 
Ethical: Results must not be framed as prescribing different treatments by sex without clinical evidence. Sex-differential findings are hypotheses for follow-up, not recommendations.

