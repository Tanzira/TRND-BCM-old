#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:45:37 2022

@author: tanzira
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.io import loadmat
root_path = './'
#%% Reading raw files
#getting input for metastatic or non metastatic coefficient calculation.
meta_val = 0 #1 means metastatic 0 means non metastatic
data_raw  = loadmat(root_path + 'Dataset/ACESExpr.mat')['data']
p_type = loadmat(root_path + 'Dataset/ACESLabel.mat')['label']
entrez_id = loadmat(root_path + 'Dataset/ACES_EntrezIds.mat')['entrez_ids']
expr_data = pd.DataFrame(data_raw)
expr_data.columns = entrez_id.reshape(-1)

#Getting only patients with metastatic/non metastatic cancer for coefficient calculation.
patients = expr_data.loc[p_type == meta_val, :]

#%% Reading TF file and getting common TF bettween gene expression and TF file
tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[human_tfs['Is TF?'] =='Yes']
human_tfs.set_index('EntrezGene ID', inplace = True)
print(human_tfs)

common_tf = np.intersect1d(expr_data.columns.astype('str'), human_tfs.index).tolist()
common_tf = list(map(int, common_tf))
tf_df = patients.loc[:, common_tf]

#%% Doing lasso here
def do_lasso(sampled_data, tf_with_exp, alpha_val):
    coefficients = []
    i = 0
    for gene in sampled_data.columns:
        selected_lung = sampled_data.copy()
        y = selected_lung.loc[:, gene].values.copy()
        if gene in tf_with_exp:
            selected_lung.loc[:, gene] = 0
        tf_exp = selected_lung.loc[:, tf_with_exp].copy()
        x = tf_exp.to_numpy()
        reg = linear_model.Lasso(alpha=alpha_val, max_iter=10000, random_state = 0)#Lasso regression model
        reg.fit(x, y)  
        coefficients.append(reg.coef_)
        i += 1

    return coefficients
#%%Calculating the coefficients here by calling the do_lasso function
alpha_val = 0.1
can_type = ['non_meta', 'meta']
csv_file = root_path + 'lasso_coef_with_alpha_{0}_{1}.csv'.format(alpha_val, can_type[meta_val])
tf_with_exp = tf_df.columns.tolist()
coefficients = do_lasso(patients, tf_with_exp, alpha_val)
coef_df = pd.DataFrame(coefficients,  index= patients.columns, columns = tf_with_exp)
coef_df.to_csv(csv_file)

