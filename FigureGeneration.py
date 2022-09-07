#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:37:30 2022

@author: tanzira
"""


#%%All Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
root_path = './'
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

root_path = './'

#%% Read ACES data
'''AECS data'''
aces_raw  = loadmat(root_path + 'Dataset/ACESExpr.mat')['data']
aces_p_type = loadmat(root_path + 'Dataset/ACESLabel.mat')['label']
aces_entrez_id = loadmat(root_path + 'Dataset/ACES_EntrezIds.mat')['entrez_ids']
aces_data = pd.DataFrame(aces_raw)
aces_data.columns = aces_entrez_id.reshape(-1)
#Reading TF file
tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[(human_tfs['Is TF?'] =='Yes') & (human_tfs['EntrezGene ID'] != 'None')]
human_tfs.set_index('EntrezGene ID', inplace = True)
human_tfs.index = human_tfs.index.astype(int)

#%% R2 plot 1010 TF
alphas = [2.0, 1.0, 0.5, 0.25, 0.13, 0.06, 0.03, 0.01, 0.008, 0.004]
df_1010 = pd.DataFrame(columns = alphas, index = aces_data.columns)

fig, axes = plt.subplots(figsize = (6, 4), constrained_layout = True)
for alpha in alphas:

    file_name = root_path + 'R2CVScores/cv_r2_score_alpha_'+ str(alpha) + '.txt'
    with open(file_name, 'r') as file:
        scores = np.array(file.read().strip().split(), dtype = np.float)
        df_1010.loc[:, alpha] = scores
ax = sns.boxplot(data = df_1010, ax = axes, showfliers = True)
plt.xlabel(r'$\lambda$ thresholds')
plt.ylabel(r'$R^2$ values')
ax.set_xticklabels([r'$2^{' + str(i) + '}$' for i in range(1, -9, -1)])
ax.invert_xaxis()
# plt.savefig('./r2_box_plot_bc.eps')
plt.show()

#%% Calculating AUC kappa
def calculate_auc_kappa(all_true, all_predictions, all_scores):
    c_test = all_true
    y_pred = all_predictions
    ds = all_scores
    acc_score = accuracy_score(c_test, y_pred)
    f1 = f1_score(c_test, y_pred)
    print("Accuracy score: {0:.3f}% and f1 score: {1:.2f}".format(acc_score*100, f1))
    # print("Classification Report: \n", classification_report(c_test, y_pred))
    
    fpr, tpr, thresholds = roc_curve(c_test, ds, drop_intermediate = False)
    auroc = roc_auc_score(c_test, ds)
    print('AUCROC:{0:3f}'.format(auroc))
    ds_reshaped = np.tile(ds, [len(thresholds), 1])
    thres_reshaped = np.reshape(thresholds, [-1, 1])
    
    cks = round(cohen_kappa_score(y_pred, c_test), 2)
    print("Final Kappa Score: ", cks)
    c_pred_thres = np.array(ds_reshaped >= thres_reshaped, dtype = int)
    c_test_thres = np.tile(c_test, [len(thresholds), 1])
    TP = ((c_pred_thres == 1) & (c_test_thres == 1)).sum(axis = 1)
    FP = ((c_pred_thres == 1) & (c_test_thres == 0)).sum(axis = 1)
    TN = ((c_pred_thres == 0) & (c_test_thres == 0)).sum(axis = 1)
    FN = ((c_pred_thres == 0) & (c_test_thres == 1)).sum(axis = 1)
    kappa = 2 * (TP*TN - FN*FP) / ((TP + FP) * (FP + TN) + (TP + FN) * (FN + TN))
    auroc = roc_auc_score(c_test, ds)
    ty = TP / (TP + FN)
    tx = FP / (FP + TN)
    return fpr, tpr, tx, ty, auroc, kappa


p_calss = pd.read_csv(root_path + 'Dataset/IntermediaryFiles/predicted_calss_for_10_fold_cv_v4.csv')
p_scores = pd.read_csv(root_path + 'Dataset/IntermediaryFiles/predicted_scores_for_10_fold_cv_v4.csv')

fpr_lasso, tpr_lasso, tx_lasso, ty_lasso, auroc_lasso, kappa_lasso = \
                calculate_auc_kappa(p_calss['all_true'], p_calss['DysRegScore'], p_scores['DysRegScore'])
fpr_rf, tpr_rf, tx_rf, ty_rf, auroc_rf, kappa_rf = \
                calculate_auc_kappa(p_calss['all_true'], p_calss['RF'], p_scores['RF'].values)

fpr_logr, tpr_logr, tx_logr, ty_logr, auroc_logr, kappa_logr = \
                calculate_auc_kappa(p_calss['all_true'], p_calss['LogisticR'], p_scores['LogisticR'].values)
fpr_dtc, tpr_dtc, tx_dtc, ty_dtc, auroc_dtc, kappa_dtc = \
                calculate_auc_kappa(p_calss['all_true'], p_calss['DecTree'], p_scores['DecTree'])
fpr_knnc, tpr_knnc, tx_knnc, ty_knnc, auroc_knnc, kappa_knnc = \
                calculate_auc_kappa(p_calss['all_true'], p_calss['KNN'], p_scores['KNN'])
fpr_mlpc, tpr_mlpc, tx_mlpc, ty_mlpc, auroc_mlpc, kappa_mlpc = \
                calculate_auc_kappa(p_calss['all_true'], p_calss['MLP'], p_scores['MLP'])
fpr_svcl, tpr_svcl, tx_svcl, ty_svcl, auroc_svcl, kappa_svcl = \
                calculate_auc_kappa(p_calss['all_true'], p_calss['SVC'], p_scores['SVC'])




#%% COMBINE AUC/KAPPA/LOSO
'''Run Calculating AUC kappa cell first'''


fig, axes = plt.subplots(figsize=(13, 3), nrows = 1, ncols = 5,
                         gridspec_kw = {"width_ratios": [5,5,2,3,5]})
ax = axes[0]
ax.plot([0, 1], [0, 1], c = '#ccc', linewidth = 0.5)

plot_data = {'DysRegScore': (fpr_lasso, tpr_lasso, auroc_lasso, max(kappa_lasso)),
             'RF': (fpr_rf, tpr_rf, auroc_rf, max(kappa_rf)),
             'MLP': (fpr_mlpc, tpr_mlpc, auroc_mlpc, max(kappa_mlpc)),
             'KNN': (fpr_knnc, tpr_knnc,auroc_knnc,  max(kappa_knnc)),
             'LogisticR': (fpr_logr, tpr_logr, auroc_logr, max(kappa_logr)),
             'DecTree': (fpr_dtc, tpr_dtc, auroc_dtc, max(kappa_dtc)),
             'SVC': (fpr_svcl, tpr_svcl, auroc_svcl, max(kappa_svcl))}

for name, data in plot_data.items():
    fpr, tpr, auroc, kappa = data
    h = ax.plot(fpr, tpr, label = '{} ({:.2f} / {:.2f})'.format(name, auroc, kappa),
             linewidth = 2 if name == 'DysRegScore' else 1,
             linestyle = '-' if name == 'DysRegScore' else '--')
handles, _ = ax.get_legend_handles_labels()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('False Positive rate')
ax.set_ylabel('True Positive rate')
#ax.set_title("ROC/AUC")
ax.text(0.5, -0.465, '(a)', ha = 'center', fontsize = 16,
        transform = ax.transAxes)
ax.legend(handles = handles,
          title = 'Model (AUROC / Kappa)',
          bbox_to_anchor = (-0.2, 1), loc = 'upper right',
          frameon = False)
#ax.set_title("(a)", fontsize = 16, pad = 10)


#plt.subplot(222)
ax = axes[1]
ax.plot(tx_lasso, kappa_lasso, label = 'DysRegScore', linewidth = 2)
ax.plot(tx_rf, kappa_rf, label = 'RF', linestyle = '--', linewidth = 1)
ax.plot(tx_mlpc, kappa_mlpc,  label = 'MLP', linestyle = '--', linewidth = 1)
ax.plot(tx_knnc, kappa_knnc,  label = 'KNN', linestyle = '--', linewidth = 1)
ax.plot(tx_logr, kappa_logr, label = 'LogisticR', linestyle = '--', linewidth = 1)
ax.plot(tx_dtc, kappa_dtc, label = 'DecTree', linestyle = '--', linewidth = 1)
ax.plot(tx_svcl, kappa_svcl, label = 'SVC', linestyle = '--', linewidth = 1)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_ylabel('Kappa')
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
#ax.yaxis.set_visible(False)

#plt.legend()

ax.set_xlabel('False Positive rate')
#ax.set_title("(b)", fontsize = 16, pad = 10)
ax.text(0.5, -0.465, '(b)', ha = 'center', fontsize = 16,
        transform = ax.transAxes)
#plt.savefig('./UpdatedFigures/AUCComparisonWithOtherModels10Fold.eps', bbox_inches = 'tight', dpi = 300)

n_iter = 50

#auc_scores = pd.read_csv('./IntermediaryFiles/AUC_scores_for_different_models_for50_iteration.csv', index_col=0)
auc_scores = pd.read_csv('./IntermediaryFiles/stratified_AUC_scores_for_different_models_for_50_iteration_v4.csv', index_col=0)

auc_scores = auc_scores.loc[:, ['DysRegScore',
                                'RF',
                                'MLP',
                                'KNN',
                                'LogisticR',
                                'DecTree',
                                'SVC']]

# padding ax
ax = axes[2]
ax.axis('off')

tick_rotation = 45
#fig, axes = plt.subplots(figsize = (8,3), ncols = 2, sharey = True)
ax = axes[3]
sns.barplot(data = auc_scores, ci = "sd", ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = tick_rotation, ha = 'right')
ax.set_ylim([0.4,1])
ax.set_ylabel('AUC', labelpad = 4)
ax.set_xlabel('(c)', labelpad = 4, fontsize = 16)
heights = auc_scores.max(axis = 0).values


# LOSO bar for 3 methods with 50 iterations
niter = 50
ax = axes[4]
loso_3_method = pd.read_csv('./IntermediaryFiles/loso_3_methods_50_iteration_v4.csv', index_col=0)
loso_3_method.iloc[:, [1, 0, 2]].plot.bar(ax = ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation = tick_rotation, ha = 'right')
ax.set_ylim([0.4,1])
ax.set_xlabel('(d)', labelpad = 15, fontsize = 16)
#ax.yaxis.set_visible(False)
ax.legend(bbox_to_anchor = [0, 1], loc = 'upper left',
          frameon = False)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.subplots_adjust(wspace = 0.2)

# plt.savefig('./performance.eps', bbox_inches = 'tight', dpi = 300)
plt.show()

#%%Edge Stats figure with only positive negative correlation cutoff V3

meta_y = pd.read_csv(root_path + 'Dataset/lasso_coef_with_alpha_0.1_meta.csv', index_col = 0)
meta_n = pd.read_csv(root_path + 'Dataset/lasso_coef_with_alpha_0.1_non_meta.csv', index_col = 0)
r2 = pd.read_csv(root_path + 'Dataset/cv_r2_score_alpha_0.1.txt', header = None, names = ['r2'])
r2.index = meta_y.index
r2 = r2.squeeze()

# prepare stats
corr_th = 0.3
coef_th = 0.0
r2_threshold = r2.values.min() - 1 
min_tg_th = 5

#for r2_threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:

goodGenes = (r2.values >= r2_threshold)
filtered_meta = meta_y.loc[goodGenes, :] # filtering genes with better r2
filtered_nmeta = meta_n.loc[goodGenes, :] # filtering genes with better r2

#keeping TFs with at least min_tg_th targets
nTargets_meta = (filtered_meta != 0).sum(axis = 0) >= min_tg_th
nTargets_nmeta = (filtered_nmeta != 0).sum(axis = 0) >= min_tg_th

#keeping TFs with at least min_tg_th targets
filtered_meta = filtered_meta.T[filtered_meta.astype(bool).sum(axis=0) >= min_tg_th].T
filtered_nmeta = filtered_nmeta.T[filtered_nmeta.astype(bool).sum(axis=0) >= min_tg_th].T

#calculating pearson correlation among TFs
meta_coreg = filtered_meta.corr()
nmeta_coreg = filtered_nmeta.corr()

#filling null values with 0
meta_coreg.fillna(0, inplace = True)
nmeta_coreg.fillna(0, inplace = True)

np.fill_diagonal(meta_coreg.values, 0) # Filling the diagonals with 0
np.fill_diagonal(nmeta_coreg.values, 0) # Filling the diagonals with 0

total_edges_m_p = (meta_coreg > 0).sum().sum()
total_edges_nm_p = (nmeta_coreg > 0).sum().sum()

total_edges_m_n = (meta_coreg < 0).sum().sum()
total_edges_nm_n = (nmeta_coreg < 0).sum().sum()


n_tf_meta = meta_coreg.shape[0]
n_tf_nmeta = nmeta_coreg.shape[0]

stats = []
p_cutoffs = np.arange(0.1, 0.5, 0.025)
for t in p_cutoffs:
    a = round(np.sum(nmeta_coreg.values >= t)*100/total_edges_nm_p, 2)
    b = round(np.sum(meta_coreg.values >= t)*100/total_edges_m_p, 2)
    c = round(np.sum(nmeta_coreg.values <= -t)*100/total_edges_nm_n, 2)
    d = round(np.sum(meta_coreg.values <= -t)*100/total_edges_m_n, 2)
    stats.append([t, a, b, c, d])
        # print(t, '\t', a, '\t', b,'\t', c,'\t',  d)
stats = (pd
         .DataFrame(stats, columns = ['cc', 'nm+', 'm+', 'nm-', 'm-'])
         .set_index('cc')
         .stack()
         .reset_index())
stats.columns = ['cc', 'type', 'percent']
stats['sign'] = stats['type'].str[-1]
stats['class'] = stats['type'].apply(lambda x: 'metastatic' if x.startswith('m') else 'non-metastatic')
stats = stats.set_index(['sign', 'cc', 'class'])['percent']

stats = stats.unstack(level = -1).sort_index().iloc[:, ::-1]
stats = stats.reset_index(level = -1)

cutoff_ntargets = 5
cutoff_r2s = [0.1, 0.2, 0.3]

nbins = 20
logscale = False

fig, axes = plt.subplots(figsize=(6, 4), nrows = 2)

ax = axes[0]
stats.loc['+', :].set_index('cc').sort_index().plot(ax = ax, style = ['-o', '-o'])
ax.set_ylabel(None)
ax.xaxis.set_visible(False)
ax.set_ylabel('% +ve correlated\nedges', labelpad = 9)
ax.legend(frameon = False, title = 'Co-regulatory net')

ax = axes[1]
stats.loc['-', :].set_index('cc').sort_index().plot(ax = ax, style = ['-o', '-o'])
ax.set_ylabel(None)
ax.set_ylabel('% -ve correlated\nedges')
ax.get_legend().remove()
ax.set_xlabel('Absolute corelation cutoff')

plt.subplots_adjust(wspace = 0.03, hspace = 0.1)

# plt.savefig('edge_stats.eps', bbox_inches = 'tight', dpi = 300)
plt.show()
