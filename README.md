# Transcriptional-regulatory-network-dysregulation-predicts-breast-cancer-metastasis

# Dataset
## All the intermediary files and datasets can be found here.
https://drive.google.com/drive/folders/1c2mL9j3MHGP8VReEtjGwChknLTjiO3UB?usp=sharing

## CoefficientCalculationLassoForGRN.py
Calculate two coefficient files for metastatic and non metastatic cancer patients. 

## TFTFCoregulatoryNet.py
- Read the coefficient files and generates networks
- Generate .gml file for cytoscape to visualize the networks
- Perform statistical analysis

## LassoCrossValidationForR2Calc.py
Generate $R^2$ for a range of $\lambda$ values

## GettingTheFiles.py
- Generate ensembl models for classification using both stratified k-fold cross-validation and leave-one-study-out cross-validation
- Save the model and intermediary files

## FigureGeneration.py
- Generate figures provided in the paper using saved intermediary files

