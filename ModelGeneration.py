#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 17:12:12 2022

@author: tanzira
"""

#%%All imports
import pandas as pd
import numpy as np
from scipy.io import loadmat
import pickle
from sklearn import linear_model
import os
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
root_path = './'
#%%Read ACES data for all operations
data_raw  = loadmat(root_path + 'Dataset/ACESExpr.mat')['data']
p_type = loadmat(root_path + 'Dataset/ACESLabel.mat')['label']
entrez_id = loadmat(root_path + 'Dataset/ACES_EntrezIds.mat')['entrez_ids']
expr_data = pd.DataFrame(data_raw)
expr_data.columns = entrez_id.reshape(-1)

tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[human_tfs['Is TF?'] =='Yes']
human_tfs.set_index('EntrezGene ID', inplace = True)

common_tf = np.intersect1d(expr_data.columns.astype('str'), human_tfs.index).tolist()
common_tf = list(map(int, common_tf))
tf_locs = [expr_data.columns.get_loc(c) for c in common_tf]



#%%Builiding models with lasso using stratified 10 fold CV
'''Lasso regression'''
def do_lasso(sampled_data, tf_locs, alpha_val):
    coefficients = []
    for loc in range(sampled_data.shape[1]):
        temp = sampled_data.copy()
        gene_exp = temp[:, loc].copy()#gene expression
        if loc in tf_locs:
            temp[:, loc] = 0
        tf_exp = temp[:, tf_locs].copy()#TFs expression
        # print(tf_exp.shape, gene_exp.shape)
        reg = linear_model.Lasso(alpha=alpha_val, max_iter=10000, random_state = 42)#Lasso regression model
        reg.fit(tf_exp, gene_exp)  
        coefficients.append(reg.coef_)
    return coefficients


#Here the model is built using 4 lambdas only that were optimal
alphas = [0.13, 0.06, 0.03]

# n_fold = int(sys.argv[1])
# cv_idx = int(sys.argv[2])
# can_type = int(sys.argv[3])

n_fold = 10
# cv_idx = 0
can_type = 1
for cv_idx in range(n_fold):
    print(cv_idx)
    #doing stratified K fold
    skf = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
    
    data = expr_data.to_numpy()
    p_type = p_type.ravel()
    cv_indices = list(skf.split(data, p_type))
    
    train_index, test_index = cv_indices[cv_idx]
    
    
    data_train, data_test, c_train, c_test = data[train_index], data[test_index], p_type[train_index], p_type[test_index]
    
    data_train = data_train[c_train == can_type, :]
    
    print(can_type, data_train.shape)
    models = []
    #Doing lasso for 10 different alphas
    for alpha_val in alphas:
        coef = do_lasso(data_train, tf_locs, alpha_val)
        models.append(coef)
    
    #Combining the models from different alpha values
    models = np.swapaxes(np.stack(models), 1, 2)
    
    
    dir_name = './MetaNonMetaModelsWithStratifiedKFold/Folds_' + str(n_fold)
    try:
        os.makedirs(dir_name, exist_ok=True)
    except OSError:
        print("Error occured")
    
    if can_type == 1:
        f1 = dir_name + '/meta_models_with_diff_alpha_cv_{}.pkl'.format(cv_idx)
    else:
        f1 = dir_name + '/nmeta_models_with_diff_alpha_cv_{}.pkl'.format(cv_idx)
    out = open(f1, 'wb')
    pickle.dump(models, out)
    out.close()

#%% Calculating AUC for models using stratified 10 fold cross validation
data = expr_data.to_numpy()
alphas = [2, 1, 0.5, 0.25, 0.13, 0.06, 0.03, 0.01, 0.008, 0.004]
n_fold = 10
cv = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
p_type = p_type.ravel()

dir_name = './MetaNonMetaModelsWithStratifiedKFoldV2/Folds_10'
l_pred_lasso, l_pred_rf, l_pred_logr, l_pred_dtc, l_pred_knnc, l_pred_mlpc, l_pred_svcl = [], [], [], [], [], [], []
all_true = []
l_scr_lasso, l_scr_rf, l_scr_logr, l_scr_dtc, l_scr_knnc, l_scr_mlpc, l_scr_svcl = [], [], [], [], [], [], []
auc_lasso, auc_rf, auc_logr, auc_dtc, auc_knnc, auc_mlpc, auc_svcl = [], [], [], [], [], [], []
idx = -1
for train_index, test_index in cv.split(data, p_type):
    idx += 1
    print('CV_IDX: ', idx)
    data_train, data_test, c_train, c_test = data[train_index], data[test_index],\
                                                p_type[train_index], p_type[test_index]

    
    f1 = dir_name + '/meta_models_with_diff_alpha_cv_' + str(idx) + '.pkl'
    f2 = dir_name + '/nmeta_models_with_diff_alpha_cv_' + str(idx) + '.pkl'
    with open(f1,'rb') as f:
        models_meta = pickle.load(f)
    with open(f2,'rb') as f11:
        models_nmeta = pickle.load(f11)
    print(models_meta.shape)
    x_test = data_test[:, tf_locs] # expression levels of TFs
    n_iter = models_meta.shape[0]

    exp_pred_1 = np.tile(x_test, [n_iter, 1, 1]) @ models_meta # predicted expression levels of all genes using class = metastatic models
    exp_pred_0 = np.tile(x_test, [n_iter, 1, 1]) @ models_nmeta # predicted expression levels of all genes using class = non-metastatic models

    avg_pred_1 = exp_pred_1.mean(axis =0)
    avg_pred_0 = exp_pred_0.mean(axis =0)
    
    # calculate euclidean scores for each model
    s1 = np.linalg.norm(data_test - avg_pred_1, axis = 1)
    s0 = np.linalg.norm(data_test - avg_pred_0, axis = 1)
                      
    m1 = RandomForestClassifier(random_state = 42)
    m2 = LogisticRegression(solver='liblinear', random_state=42)
    m3 = DecisionTreeClassifier(random_state = 42)
    m4 = KNeighborsClassifier()#No random state in parameter
    m5 = MLPClassifier(random_state=42)#Neural network classifier
    m6 = svm.SVC(kernel = 'linear', random_state=42, probability=True)
    
    m1.fit(data_train, c_train)
    m2.fit(data_train, c_train)
    m3.fit(data_train, c_train)
    m4.fit(data_train, c_train)
    m5.fit(data_train, c_train)
    m6.fit(data_train, c_train)
    
    pred_lasso = s0-s1
    y_pred_lasso = [int(l > 0) for l in pred_lasso]
    
    pred_rf = m1.predict_proba(data_test)[:, 1]
    y_pred_rf = m1.predict(data_test)
    
    pred_logr = m2.predict_proba(data_test)[:, 1]
    y_pred_logr = m2.predict(data_test)
    
    pred_dtc = m3.predict_proba(data_test)[:, 1]
    y_pred_dtc = m3.predict(data_test)
    
    pred_knnc = m4.predict_proba(data_test)[:, 1]
    y_pred_knnc = m4.predict(data_test)
    
    pred_mlpc = m5.predict_proba(data_test)[:, 1]
    y_pred_mlpc = m5.predict(data_test)
    
    pred_svcl = m6.predict_proba(data_test)[:, 1]
    y_pred_svcl = m6.predict(data_test)
    
    l_pred_lasso.extend(y_pred_lasso)
    l_scr_lasso.extend(pred_lasso)
    auc_lasso.append(roc_auc_score(c_test, pred_lasso))
    
    l_pred_rf.extend(y_pred_rf)
    l_scr_rf.extend(pred_rf)
    auc_rf.append(roc_auc_score(c_test, pred_rf))
    
    l_pred_logr.extend(y_pred_logr)
    l_scr_logr.extend(pred_logr)
    auc_logr.append(roc_auc_score(c_test, pred_logr))
    
    l_pred_dtc.extend(y_pred_dtc)
    l_scr_dtc.extend(pred_dtc)
    auc_dtc.append(roc_auc_score(c_test, pred_dtc))
    
    l_pred_knnc.extend(y_pred_knnc)
    l_scr_knnc.extend(pred_knnc)
    auc_knnc.append(roc_auc_score(c_test, pred_knnc))
    
    l_pred_mlpc.extend(y_pred_mlpc)
    l_scr_mlpc.extend(pred_mlpc)
    auc_mlpc.append(roc_auc_score(c_test, pred_mlpc))
    
    l_pred_svcl.extend(y_pred_svcl)
    l_scr_svcl.extend(pred_svcl)
    auc_svcl.append(roc_auc_score(c_test, pred_svcl))

    all_true.extend(c_test)

#stratified class and score saving
columns =  ['DysRegScore', 'RF', 'LogisticR', 'DecTree', 'KNN', 'MLP', 'SVC', 'all_true']

predicted_calss = pd.DataFrame( 0, index = np.arange(len(l_pred_lasso)), columns = columns)
predicted_calss['DysRegScore'] = l_pred_lasso
predicted_calss['RF'] = l_pred_rf
predicted_calss['LogisticR'] = l_pred_logr
predicted_calss['DecTree'] = l_pred_dtc
predicted_calss['KNN'] = l_pred_knnc
predicted_calss['MLP'] = l_pred_mlpc
predicted_calss['SVC'] = l_pred_svcl
predicted_calss['all_true'] = all_true

#Uncomment if you want to save the file
# predicted_calss.to_csv('./predicted_calss_for_10_fold_cv_v4.csv')


columns =  ['DysRegScore', 'RF', 'LogisticR', 'DecTree', 'KNN', 'MLP', 'SVC']

predicted_scores = pd.DataFrame( 0, index = np.arange(len(l_scr_lasso)), columns = columns)
predicted_scores['DysRegScore'] = l_scr_lasso
predicted_scores['RF'] = l_scr_rf
predicted_scores['LogisticR'] = l_scr_logr
predicted_scores['DecTree'] = l_scr_dtc
predicted_scores['KNN'] = l_scr_knnc
predicted_scores['MLP'] = l_scr_mlpc
predicted_scores['SVC'] = l_scr_svcl

#Uncomment if you would like to save the calculations. Other models takes time to compute everything specially MLP
# predicted_scores.to_csv('./predicted_scores_for_10_fold_cv_v4.csv')
#%% AUC calculation 50 Stratified
data_raw  = loadmat(root_path + 'ACES_Data/ACESExpr.mat')['data']
p_type = loadmat(root_path + 'ACES_Data/ACESLabel.mat')['label']
entrez_id = loadmat(root_path + 'ACES_Data/ACES_EntrezIds.mat')['entrez_ids']
expr_data = pd.DataFrame(data_raw)
expr_data.columns = entrez_id.reshape(-1)

tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[human_tfs['Is TF?'] =='Yes']
human_tfs.set_index('EntrezGene ID', inplace = True)

common_tf = np.intersect1d(expr_data.columns.astype('str'), human_tfs.index).tolist()
common_tf = list(map(int, common_tf))
tf_locs = [expr_data.columns.get_loc(c) for c in common_tf]

data = expr_data.to_numpy()
p_type = p_type.ravel()

alphas = [2, 1, 0.5, 0.25, 0.13, 0.06, 0.03, 0.01, 0.008, 0.004]
n_fold = 10

cv = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)

dir_name = './MetaNonMetaModelsWithStratifiedKFold/Folds_10'
niter = 50
auc_scores = pd.DataFrame( 0, index = np.arange(niter), columns = ['DysRegScore', 'RF', 'LogisticR', 'DecTree', 'KNN', 'MLP', 'SVC'])
for i in range(niter):
    # l_pred_lasso, l_pred_rf, l_pred_logr, l_pred_dtc, l_pred_knnc, l_pred_mlpc, l_pred_svcl = [], [], [], [], [], [], []
    all_true = []
    scores_lasso, scores_rf, scores_logr, scores_dtc, scores_knnc, scores_mlpc, scores_svcl = [], [], [], [], [], [], []
    # auc_lasso, auc_rf, auc_logr, auc_dtc, auc_knnc, auc_mlpc, auc_svcl = [], [], [], [], [], [], []
    idx = -1
    print("iteration: ", i)
    for train_index, test_index in cv.split(data, p_type):
        idx += 1
        print('CV_IDX: ', idx)
        data_train, data_test, c_train, c_test = data[train_index], data[test_index],\
            p_type[train_index], p_type[test_index]
        data_train_meta = data_train[c_train == 1, :]
        data_train_nmeta = data_train[c_train == 0, :]
        
        chosen_idx = np.random.choice(len(data_test), replace=False, size=int(len(data_test)*0.8))
        data_test, c_test = data_test[chosen_idx], c_test[chosen_idx]
        
        f1 = dir_name + '/meta_models_with_diff_alpha_cv_' + str(idx) + '.pkl'
        f2 = dir_name + '/nmeta_models_with_diff_alpha_cv_' + str(idx) + '.pkl'
        with open(f1,'rb') as f:
            models_meta = pickle.load(f)
        with open(f2,'rb') as f11:
            models_nmeta = pickle.load(f11)
            
        models_meta = models_meta[4:-2]
        models_nmeta = models_nmeta[4:-2]
        print(models_meta.shape)
        x_test = data_test[:, tf_locs] # expression levels of TFs
        n_iter = models_meta.shape[0]
    
        exp_pred_1 = np.tile(x_test, [n_iter, 1, 1]) @ models_meta # predicted expression levels of all genes using class = metastatic models
        exp_pred_0 = np.tile(x_test, [n_iter, 1, 1]) @ models_nmeta # predicted expression levels of all genes using class = non-metastatic models
    
        avg_pred_1 = exp_pred_1.mean(axis =0)
        avg_pred_0 = exp_pred_0.mean(axis =0)
        
        # calculate euclidean scores for each model
        s1 = np.linalg.norm(data_test - avg_pred_1, axis = 1)
        s0 = np.linalg.norm(data_test - avg_pred_0, axis = 1)
                          
        m1 = RandomForestClassifier(random_state = None)
        m2 = LogisticRegression(solver='liblinear', random_state=None)
        m3 = DecisionTreeClassifier(random_state = None)
        m4 = KNeighborsClassifier()#No random state in parameter
        m5 = MLPClassifier(random_state=None)#Neural network classifier
        m6 = svm.SVC(kernel = 'linear', random_state=None, probability=True)
        
        m1.fit(data_train, c_train)
        m2.fit(data_train, c_train)
        m3.fit(data_train, c_train)
        m4.fit(data_train, c_train)
        m5.fit(data_train, c_train)
        m6.fit(data_train, c_train)
        
        pred_lasso = s0-s1
        
        y_pred_lasso = [int(l >= 0) for l in pred_lasso]
        
        pred_rf = m1.predict_proba(data_test)[:, 1]
        # y_pred_rf = m1.predict(data_test)
        
        pred_logr = m2.predict_proba(data_test)[:, 1]
        # y_pred_logr = m2.predict(data_test)
        
        pred_dtc = m3.predict_proba(data_test)[:, 1]
        # y_pred_dtc = m3.predict(data_test)
        
        pred_knnc = m4.predict_proba(data_test)[:, 1]
        # y_pred_knnc = m4.predict(data_test)
        
        pred_mlpc = m5.predict_proba(data_test)[:, 1]
        # y_pred_mlpc = m5.predict(data_test)
        
        pred_svcl = m6.predict_proba(data_test)[:, 1]
        # y_pred_svcl = m6.predict(data_test)
        
        # l_pred_lasso.extend(y_pred_lasso)
        scores_lasso.extend(pred_lasso)
        # auc_lasso.append(roc_auc_score(c_test, pred_lasso))
        
        # l_pred_rf.extend(y_pred_rf)
        scores_rf.extend(pred_rf)
        # auc_rf.append(roc_auc_score(c_test, pred_rf))
        
        # l_pred_logr.extend(y_pred_logr)
        scores_logr.extend(pred_logr)
        # auc_logr.append(roc_auc_score(c_test, pred_logr))
        
        # l_pred_dtc.extend(y_pred_dtc)
        scores_dtc.extend(pred_dtc)
        # auc_dtc.append(roc_auc_score(c_test, pred_dtc))
        
        # l_pred_knnc.extend(y_pred_knnc)
        scores_knnc.extend(pred_knnc)
        # auc_knnc.append(roc_auc_score(c_test, pred_knnc))
        
        # l_pred_mlpc.extend(y_pred_mlpc)
        scores_mlpc.extend(pred_mlpc)
        # auc_mlpc.append(roc_auc_score(c_test, pred_mlpc))
        
        # l_pred_svcl.extend(y_pred_svcl)
        scores_svcl.extend(pred_svcl)
        # auc_svcl.append(roc_auc_score(c_test, pred_svcl))
    
        all_true.extend(c_test)
    #calculating AUC for different models

    auc_lasso, auc_rf, auc_logr, auc_dtc, auc_knnc, auc_mlpc, auc_svcl = roc_auc_score(all_true, scores_lasso),\
                roc_auc_score(all_true, scores_rf), roc_auc_score(all_true, scores_logr),\
                roc_auc_score(all_true, scores_dtc), roc_auc_score(all_true, scores_knnc),\
                roc_auc_score(all_true, scores_mlpc), roc_auc_score(all_true, scores_svcl)
    auc_scores.iloc[i] = [auc_lasso, auc_rf, auc_logr, auc_dtc, auc_knnc, auc_mlpc, auc_svcl]

auc_scores.to_csv('stratified_AUC_scores_for_different_models_for_{}_iteration_v4.csv'.format(niter))

#%% Building the LOSO models

'''Takes time to build the models'''

cv_train_idx_file = root_path + 'Dataset/CVIndTrain200.txt'

train_cv_idx = pd.read_csv(cv_train_idx_file, header = None, sep = ' ')
d_map = pd.DataFrame(0, 
                     index = range(train_cv_idx.shape[0]),
                     columns = range(train_cv_idx.shape[1]))

for col in train_cv_idx.columns:
    idx_other = train_cv_idx[col][train_cv_idx[col] > 0]-1
    idx = np.setdiff1d(range(train_cv_idx.shape[0]), idx_other)
    d_map.loc[idx, col] = 1

def do_lasso(sampled_data, tf_locs, alpha_val):
    coefficients = []
    for loc in range(sampled_data.shape[1]):
        temp = sampled_data.copy()
        gene_exp = temp[:, loc].copy()#gene expression
        if loc in tf_locs:
            temp[:, loc] = 0
        tf_exp = temp[:, tf_locs].copy()#TFs expression
        # print(tf_exp.shape, gene_exp.shape)
        reg = linear_model.Lasso(alpha=alpha_val, max_iter=10000)#Lasso regression model
        # reg = linear_model.ElasticNet(alpha=alpha_val, max_iter=10000)#Elastic Net model
        reg.fit(tf_exp, gene_exp)  
        coefficients.append(reg.coef_)
    return coefficients

#Built the model for multiple alphas just to check the AUC's. All the results are calculated using 4 optimal alpha
alphas = [2, 1, 0.5, 0.25, 0.13, 0.06, 0.03, 0.01, 0.008, 0.004]


# can_type = int(sys.argv[1])
# cv_idx = int(sys.argv[2])

#set cancer type and which cross validation index you want to run. I didivided it for faster computation.
can_type = 1
cv_idx = 0
n_fold = 12
data = expr_data.to_numpy()

train_index, test_index = d_map[d_map[cv_idx] == 0].index, d_map[d_map[cv_idx] == 1].index

data_train, data_test, c_train, c_test = data[train_index], data[test_index], p_type[train_index], p_type[test_index]
data_train_meta = data_train[c_train.T[0] == can_type, :]

models = []
#Doing lasso for 10 different alphas
for alpha_val in alphas:
    coefs = do_lasso(data_train_meta, tf_locs, alpha_val)

    models.append(coefs)

# #Combining the models from different alpha values
models = np.swapaxes(np.stack(models), 1, 2)


dir_name = './LassoLOSOMetaNonMetaModels/Folds_' + str(n_fold)
try:
    os.makedirs(dir_name, exist_ok=True)
except OSError:
    print("Error occured")
if can_type == 1:
    f1 = dir_name + '/meta_models_with_diff_alpha_cv_{}.pkl'.format(cv_idx)
if can_type == 0:
    f1 = dir_name + '/nmeta_models_with_diff_alpha_cv_{}.pkl'.format(cv_idx)

##Uncomment if you want to save the models
# out = open(f1, 'wb')
# pickle.dump(models, out)
# out.close()

#%%Loso Random Forest and Lasso and MLP
cv_train_idx_file = root_path + 'ACES_Data/CVIndTrain200.txt'
# cv_test_idx_file = '/home/tanzira/Documents/MetastaticNonmetastatic/ACES_Data/CVIndTest200.txt'
train_cv_idx = pd.read_csv(cv_train_idx_file, header = None, sep = ' ')
d_map = pd.DataFrame(0, 
                     index = range(train_cv_idx.shape[0]),
                     columns = range(train_cv_idx.shape[1]))

for col in train_cv_idx.columns:
    idx_other = train_cv_idx[col][train_cv_idx[col] > 0]-1
    idx = np.setdiff1d(range(train_cv_idx.shape[0]), idx_other)
    d_map.loc[idx, col] = 1
    
alphas = [2, 1, 0.5, 0.25, 0.13, 0.06, 0.03, 0.01, 0.008, 0.004]

n_fold = 12
data = expr_data.to_numpy()

all_predictions, all_true, all_scores = [], [], []

dir_name = './LassoLOSOMetaNonMetaModels/Folds_' + str(n_fold)

niter = 50
all_auc_rf, all_auc_lasso, all_auc_mlpc = [], [], []
for i in range(niter):
    auc_lasso, auc_rf, auc_mlpc = [],[],[]
    kappa_lasso, kappa_rf, kappa_mlpc = [],[],[]
    print("iteration: ", i)
    for cv_idx in range(n_fold):
        train_index, test_index = d_map[d_map[cv_idx] == 0].index, d_map[d_map[cv_idx] == 1].index
        data_train, data_test, c_train, c_test = data[train_index], data[test_index],\
                            p_type[train_index].ravel(), p_type[test_index].ravel()
        
        f1 = dir_name + '/meta_models_with_diff_alpha_cv_' + str(cv_idx) + '.pkl'
        f2 = dir_name + '/nmeta_models_with_diff_alpha_cv_' + str(cv_idx) + '.pkl'
        with open(f1,'rb') as f:
            models_meta = pickle.load(f)
        with open(f2,'rb') as f11:
            models_nmeta = pickle.load(f11)
        
        #Taking 4 middle alphas that provides better performance
        models_meta = models_meta[4:-2]
        
        models_nmeta = models_nmeta[4:-2]
        # data_train = data_train[:, genes_var_h]
        n_iter = models_meta.shape[0]
        Xtest = data_test[:, tf_locs] 
    
        test_pred_1 = np.tile(Xtest, [n_iter, 1, 1]) @ models_meta # predicted expression levels of all genes using class = metastatic models
        test_pred_0 = np.tile(Xtest, [n_iter, 1, 1]) @ models_nmeta # predicted expression levels of all genes using class = non-metastatic models
        
        avg_pred_1 = test_pred_1.mean(axis =0)
        avg_pred_0 = test_pred_0.mean(axis =0)
        
        s1 = np.linalg.norm(data_test - avg_pred_1, axis = 1)
        s0 = np.linalg.norm(data_test - avg_pred_0, axis = 1)
        
        m1 = RandomForestClassifier(random_state=None)
        m2 = MLPClassifier(random_state=None)#Neural network classifier
        
        m1.fit(data_train, c_train.ravel())
        m2.fit(data_train, c_train.ravel())
        
        pred_rf = m1.predict_proba(data_test)[:, 1]
        y_pred_rf = m1.predict(data_test)
        
        pred_mlpc = m2.predict_proba(data_test)[:, 1]
        y_pred_mlpc = m2.predict(data_test)
        
        pred_lasso = s0-s1
        y_pred_lasso = [int(l >= 0) for l in pred_lasso]
        
        auc_rf.append(roc_auc_score(c_test, pred_rf))
        kappa_rf.append(round(cohen_kappa_score(y_pred_rf, c_test), 2))
        
        auc_mlpc.append(roc_auc_score(c_test, pred_mlpc))
        kappa_mlpc.append(round(cohen_kappa_score(y_pred_mlpc, c_test), 2))
        
        auc_lasso.append(roc_auc_score(c_test, pred_lasso))
        kappa_lasso.append(round(cohen_kappa_score(y_pred_lasso, c_test), 2))
        
    all_auc_rf.append(auc_rf)
    all_auc_mlpc.append(auc_mlpc)
    all_auc_lasso.append(auc_lasso)
auc_rf = np.array(all_auc_rf)
auc_mlpc = np.array(all_auc_mlpc)
auc_lasso = np.array(all_auc_lasso)
auc_rf_mean = auc_rf.mean(axis = 0)
auc_mlpc_mean = auc_mlpc.mean(axis = 0)
auc_lasso_mean = auc_lasso.mean(axis = 0)
columns=['Desmedt', 'Hatzis', 'Ivshina', 'Loi', 'Miller', 'Minn', 'Pawitan', 'Schmidt', 'Symmans', 'WangY', 'WangYE', 'Zhang']
df = pd.DataFrame([auc_rf_mean, auc_lasso_mean, auc_mlpc_mean], index = ['RF', 'DysRegScore', 'MLP'], columns = columns).T

#Uncomment if you want to same the file
# df.to_csv('loso_3_methods_{0}_iteration_v4.csv'.format(niter))
