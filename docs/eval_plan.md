Test: if the AI models are biased? Does it make more mistakes predicting drug response in certain sex cells over others and whether extra biological data can result in equal accuracy for both sexes without getting worse overall? 


Primary endpoint: 

RMSE on held-out ln(IC50) in one declared unit (or AUC in separate experiment). 

Record target transformation, GDSC release, response aggregation policy, batch source, per-drug sample counts, and preprocessing version. 

For sex `s` metrics: 

RMSE_s: sqrt(mean((observed - predicted)^2) among observations with sex=s): Model’s average mistake size only on cells from sex s. 

Signed gap: (RMSE_Female - RMSE_Male): Allows us to quantify whether male or female cells have worse predictions and by how much. 

Absolute gap: abs(signed gap): Magnitude of signed gap

Pearson and Spearman correlations per sex; Checks whether models can rank drugs in the right order for each sex (i.e. correctly identifying which drugs are most effective) 


per-tissue RMSE: check the accuracy from each organ to organ

macro-averaged across drugs and cell lines to limit overrepresentation: some popular drugs or cell lines get tested multiple times, while rare ones only get tested once. Macro averaging calcculates the score for each individually and then averages them and helps prevent over testing 


Assess group support and performance confidence intervals: how many male and female samples exist in each group and error bars so you know if a gap is statistically significant


