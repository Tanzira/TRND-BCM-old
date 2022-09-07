# Transcriptional-regulatory-network-dysregulation-predicts-breast-cancer-metastasis

# Dataset
## All the intermediary files and datasets are uploaded here.
https://drive.google.com/drive/folders/1c2mL9j3MHGP8VReEtjGwChknLTjiO3UB?usp=sharing

## CoefficientCalculationLassoForGRN.py
  #### It calculates two coefficient files for metastatic and non metastatic cancer patients. 

## TFTFCoregulatoryNet.py
  #### This file reads the coefficient files and generates the network in different ways.
  #### It also generates some .gml file for cytoscape to visualize the networks.
  #### All the statistical analysis for the networks are also included in here.

## LassoCrossValidationForR2Calc.py
  #### This file generates $R^2$ for different lambdas that help us choosing lambdas for our models. 

## GettingTheFiles.py
  #### This one generates the bigger files for classification.
  #### There are two types of models that can be created here.
  ##### One is for stratified 10 fold cross validation
  ##### Second is for leave one study out method.
  ##### The files model generation takes a little amount of time to but once it's done classification does not take much time.
  ##### We saved the model and intermediary files using the codes here.

## FigureGeneration.py
  #### It generates the figures we got for classification using some saved intermediary files. 
  #### It also generates one figure for the network that shows the correlation trend of TF coregulatory network between two cancer types.

