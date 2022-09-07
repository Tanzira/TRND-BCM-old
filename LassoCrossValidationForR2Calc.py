#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:23:42 2022

@author: tanzira
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold
import os
from scipy.io import loadmat

root_path = './'
#%% Getting input  from ARC for divided computation
#Divided the genes into multiple chunks for faster computation. Otherwise it takes days to complete.
# n = int(sys.argv[1])
# alpha_val = float(sys.argv[2])
# chunk_val = int(sys.argv[3])



#%% Reading Raw data
data_raw  = loadmat(root_path + 'Dataset/ACESExpr.mat')['data']
p_type = loadmat(root_path + 'Dataset/ACESLabel.mat')['label']
entrez_id = loadmat(root_path + 'Dataset/ACES_EntrezIds.mat')['entrez_ids']
expr_data = pd.DataFrame(data_raw)
expr_data.columns = entrez_id.reshape(-1)

# Reading TF file
tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[human_tfs['Is TF?'] =='Yes']
human_tfs.set_index('EntrezGene ID', inplace = True)
#Getting common TFs among expression data and the TF database
common_tf = np.intersect1d(expr_data.columns.astype('str'), human_tfs.index).tolist()
common_tf = list(map(int, common_tf))
tf_df = expr_data.loc[:, common_tf]

#%%Lasso regression 10 fold cross validation
def lasso_regression(X, y):
    scores = []
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        #for lasso
        reg = linear_model.Lasso(alpha = alpha_val)
        reg.fit(X_train, y_train)  
        scores.append(reg.score(X_test, y_test))
    return np.mean(scores)
#%% Dividing genes in chunk
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out
#%%Saving R2 scores returned from lasso regression
def do_lasso(n, tf_with_exp):
    scores = []
    for gene in chunked_genes[n]:
        selected_bc = expr_data.copy()
        y = selected_bc.loc[:, gene].values.copy()
        if gene in tf_with_exp:
            selected_bc.loc[:, gene] = 0
        tf_exp = selected_bc.loc[:, tf_with_exp].copy()
        X = tf_exp.to_numpy()
        score = lasso_regression(X, y)
        scores.append(score)
    wb_score_file = dir_name + "/" + str(n) + ".txt"
    with open(wb_score_file, 'w+') as file:
        file.write('\n'.join(str(i) for i in scores))
#%% Dividing in chunks for faster computation on ARC
alpha_val = 0.1
chunk_val = 20
for n in range(chunk_val):
    dir_name = './R2CVScores/alpha_'+str(alpha_val)
    try:
        os.makedirs(dir_name, exist_ok=True)
    except OSError:
        print("Error occured")
    
    gene_list = expr_data.columns.values
    chunked_genes = chunkIt(gene_list, chunk_val)
    tf_with_exp = tf_df.columns.tolist()
    print("n and len: ", n, len(chunked_genes[n]))
    do_lasso(n, tf_with_exp)
