# Transcriptional-regulatory-network-dysregulation-predicts-breast-cancer-metastasis

# Dataset
## All the intermediary files and datasets can be found here.
https://drive.google.com/drive/folders/1c2mL9j3MHGP8VReEtjGwChknLTjiO3UB?usp=sharing
## NetworkAnalysis.py
- Generate network for a specific lambda using bootstrap
- Read the coefficient files and generates networks
- Generate .gml file for cytoscape to visualize the networks
- Perform statistical analysis

## ModelGenerationAndScoreCalculation.py
- Generate ensembl models for classification using both stratified k-fold cross-validation and leave-one-study-out cross-validation
- Generate models for NKI validation dataset.
- Save the model and intermediary score files

## FigureGeneration.py
- Generate figures provided in the paper using saved intermediary files

