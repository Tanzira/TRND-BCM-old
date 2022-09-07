#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:20:15 2022

@author: tanzira
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import hypergeom
import networkx as nx
root_path = './'

#%% Reading raw expression data and the TF file
data_raw  = loadmat(root_path + 'Dataset/ACESExpr.mat')['data']
p_type = loadmat(root_path + 'Dataset/ACESLabel.mat')['label']
entrez_id = loadmat(root_path + 'Dataset/ACES_EntrezIds.mat')['entrez_ids']
expr_data = pd.DataFrame(data_raw)
expr_data.columns = entrez_id.reshape(-1)

tf_file = 'http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.txt'
human_tfs = pd.read_csv(tf_file, sep = '\t', usecols=(1, 2, 4, 5, 11))
human_tfs = human_tfs[human_tfs['Is TF?'] =='Yes']
human_tfs.set_index('EntrezGene ID', inplace = True)

#%% Reading coefficient files
alpha_val = 0.1
can_type = ['non_meta', 'meta']
#filepath for the coefficients
csv_nmeta = root_path + 'Dataset/lasso_coef_with_alpha_{0}_{1}.csv'.format(alpha_val, can_type[0])
csv_meta = root_path + 'Dataset/lasso_coef_with_alpha_{0}_{1}.csv'.format(alpha_val, can_type[1])
#reading the coefficients
lasso_meta = pd.read_csv(csv_meta, index_col = 0)
lasso_nmeta = pd.read_csv(csv_nmeta, index_col = 0)


#%%TF-TF Coregulatory network creation 
'''Creating TF TF coregulatory and antiregulatory network'''
#Filtering lasso coef based on R2
r2 = pd.read_csv(root_path + 'Dataset/cv_r2_score_alpha_0.1.txt',header=None)
r2_threshold = 0.3
goodGenes = (r2.values >= r2_threshold)
filtered_meta = lasso_meta.loc[goodGenes, :] # filtering genes with better r2
filtered_nmeta = lasso_nmeta.loc[goodGenes, :] # filtering genes with better r2

coef_th = 0.01
filtered_meta[filtered_meta > coef_th] = 1
filtered_meta[filtered_meta < -coef_th] = -1
filtered_meta[np.abs(filtered_meta) < 1] = 0
#creating a mask for antiregulatory and coregulatory network
mask_p_meta = filtered_meta == 1
mask_n_meta = filtered_meta == -1

meta_plus = filtered_meta.copy()
meta_minus =  filtered_meta.copy()

#making all the positive relationships to 1 and the others as 0
meta_plus[mask_p_meta] = 1
meta_plus[~mask_p_meta] = 0

#making all the negative relationships to 1 and the others as 0
meta_minus[mask_n_meta] = 1
meta_minus[~mask_n_meta] = 0

filtered_nmeta[filtered_nmeta > coef_th] = 1
filtered_nmeta[filtered_nmeta < -coef_th] = -1
filtered_nmeta[np.abs(filtered_nmeta) < 1] = 0
#creating a mask for antiregulatory and coregulatory network
mask_p_nmeta = filtered_nmeta == 1
mask_n_nmeta = filtered_nmeta == -1

nmeta_plus = filtered_nmeta.copy()
nmeta_minus =  filtered_nmeta.copy()

#making all the positive relationships to 1 and the others as 0
nmeta_plus[mask_p_nmeta] = 1
nmeta_plus[~mask_p_nmeta] = 0
#making all the negative relationships to 1 and the others as 0
nmeta_minus[mask_n_nmeta] = 1
nmeta_minus[~mask_n_nmeta] = 0

#creating coregulatory and antiregulatory network for metastatic
coreg_meta = (meta_plus.T @ meta_plus) + (meta_minus.T @ meta_minus)
antireg_meta = (meta_plus.T @ meta_minus) + (meta_minus.T @ meta_plus)
np.fill_diagonal(coreg_meta.values, 0) # Filling the diagonals with 0
np.fill_diagonal(antireg_meta.values, 0) # Filling the diagonals with 0

#creating coregulatory and antiregulatory network for not metastatic
coreg_nmeta = (nmeta_plus.T @ nmeta_plus) + (nmeta_minus.T @ nmeta_minus)
antireg_nmeta = (nmeta_plus.T @ nmeta_minus)  + (nmeta_minus.T @ nmeta_plus)
np.fill_diagonal(coreg_nmeta.values, 0) # Filling the diagonals with 0
np.fill_diagonal(antireg_nmeta.values, 0) # Filling the diagonals with 0
'''Saving the TF TF coregulatory network'''
def saving_tf_tf_coreg_antireg_network(tf_tf_coreg, tf_tf_antireg, fname):
    #Different edge threshold for picking a sparser network
    edge_ths =[1, 5, 10,40, 50, 60, 70, 80]
    for edge_th in edge_ths:
        #pval_th = -np.log10(pval)
        tf_coreg = tf_tf_coreg.copy()
        tf_coreg = pd.DataFrame(tf_coreg >= edge_th, dtype = np.int)
        if isinstance(tf_tf_antireg, pd.DataFrame):
            tf_antireg = tf_tf_antireg.copy()
            tf_antireg = pd.DataFrame(tf_antireg>= edge_th, dtype = np.int)*2
            tf_net = tf_coreg + tf_antireg
        else:
            tf_net = tf_coreg
        G = nx.from_numpy_matrix(tf_net.values, create_using=None)
        label_mapping = {idx: val for idx, val in enumerate(human_tfs.loc[tf_net.columns.astype(str), 'HGNC symbol'])}
        G = nx.relabel_nodes(G, label_mapping)
        G.remove_nodes_from(list(nx.isolates(G)))
        print(nx.info(G))
        save_file = root_path + '/tf_tf_{0}_edge_th_{1}_alpha_{2}_coef_{3}_V3.gml'.\
                format(fname, edge_th, alpha_val, coef_th)
        nx.write_gml(G, save_file)
        
        
saving_tf_tf_coreg_antireg_network(coreg_meta, antireg_meta, 'meta_anti_coreg')
print('\n')
saving_tf_tf_coreg_antireg_network(coreg_nmeta, antireg_nmeta, 'nmeta_anti_coreg')
#%%Coregulatory network based on correlation 
''' The figure we created from this part is on the FigureGenerationFile.py'''


meta_y = pd.read_csv('Dataset/lasso_coef_with_alpha_0.1_meta.csv', index_col = 0)
meta_n = pd.read_csv('Dataset/lasso_coef_with_alpha_0.1_non_meta.csv', index_col = 0)
r2 = pd.read_csv('Dataset/cv_r2_score_alpha_0.1.txt', header = None, names = ['r2'])
r2.index = meta_y.index
r2 = r2.squeeze()



# prepare stats
corr_th = 0.3
coef_th = 0.0
r2_threshold = r2.values.min() - 1 

#To take TFs that has at least 5 targets in it.
min_tg_th = 5

goodGenes = (r2.values >= r2_threshold)
filtered_meta = meta_y.loc[goodGenes, :] # filtering genes with better r2
filtered_nmeta = meta_n.loc[goodGenes, :] # filtering genes with better r2

#keeping TFs with at least min_tg_th targets
nTargets_meta = (filtered_meta != 0).sum(axis = 0) >= min_tg_th
nTargets_nmeta = (filtered_nmeta != 0).sum(axis = 0) >= min_tg_th

# filtered_meta = filtered_meta.loc[:, nTargets_meta & nTargets_nmeta]
# filtered_nmeta = filtered_nmeta.loc[:, nTargets_meta & nTargets_nmeta]

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

#%%Network Topology analysis function
def network_topology(graph, singleton = True):
    avg_spl, diameter = 0, 0
    #getting number of edges
    n_edges = nx.number_of_edges(graph)
    #getting number of singletons
    n_singletons = len(list(nx.isolates(graph)))
    
    if not singleton:
        graph.remove_nodes_from(list(nx.isolates(graph)))
        #getting average shortest path
        if nx.is_connected(graph):
            avg_spl = nx.average_shortest_path_length(graph)
            #getting diameter
            diameter = nx.diameter(graph)
    #getting largest connected components
    gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
    lcc = len(graph.subgraph(gcc[0]))
    #getting average degree
    avg_deg = 2*graph.number_of_edges() / float(graph.number_of_nodes())
    #getting the largest degree
    l_deg = sorted(graph.degree, key=lambda x: x[1], reverse=True)[0][1]
    #getting average clustering coefficients
    avg_cc = nx.average_clustering(graph)
    print(nx.info(graph))
    return n_edges, n_singletons, lcc, avg_deg, l_deg, avg_cc, avg_spl, diameter
#%%TF co-regulation stats using same number of TF with different correlation threshold

r2_threshold = r2.values.min() - 1
min_tg_th = 5
stats = []

for corr_th in [0.2, 0.3, 0.4]:
    goodGenes = (r2.values >= r2_threshold)
    filtered_meta = lasso_meta.loc[goodGenes, :] # filtering genes with better r2
    filtered_nmeta = lasso_nmeta.loc[goodGenes, :] # filtering genes with better r2

   
    #keeping TFs with at least min_tg_th targets
    nTargets_meta = (filtered_meta != 0).sum(axis = 0) >= min_tg_th
    nTargets_nmeta = (filtered_nmeta != 0).sum(axis = 0) >= min_tg_th
   
    filtered_meta = filtered_meta.loc[:, nTargets_meta & nTargets_nmeta]
    filtered_nmeta = filtered_nmeta.loc[:, nTargets_meta & nTargets_nmeta]

   
    #calculating pearson correlation among TFs
    meta_coreg = filtered_meta.corr()
    nmeta_coreg = filtered_nmeta.corr()
   
    #filling null values with 0
    meta_coreg.fillna(0, inplace = True)
    nmeta_coreg.fillna(0, inplace = True)
    #setting abs(correlation) < 0.05 as 0
   
    np.fill_diagonal(meta_coreg.values, 0) # Filling the diagonals with 0
    np.fill_diagonal(nmeta_coreg.values, 0) # Filling the diagonals with 0
   
    meta_coreg[np.abs(meta_coreg) < corr_th] = 0
    nmeta_coreg[np.abs(nmeta_coreg) < corr_th] = 0
   
   
    #creating network with networkx using dataframe adj matrix
    tf_tf_GM = nx.convert_matrix.from_pandas_adjacency(meta_coreg)
    tf_tf_GNM = nx.convert_matrix.from_pandas_adjacency(nmeta_coreg)
   

    topology_col = ['metastatic', 'non_metastatic']
   
    net_topology = pd.DataFrame(0.0, columns=topology_col)
    singleton = False
    n_edges_m, n_singltn_m, lcc_m, avg_deg_m, l_deg_m, avg_cc_m, avg_spl_m, d_m = network_topology(tf_tf_GM.copy(), singleton)
    n_edges_nm, n_singltn_nm, lcc_nm, avg_deg_nm, l_deg_nm, avg_cc_nm, avg_spl_nm, d_nm = network_topology(tf_tf_GNM.copy(), singleton)
   
   
    nodes_m, nodes_nm = tf_tf_GM.number_of_nodes(), tf_tf_GNM.number_of_nodes()
    tf_tf_GM.remove_nodes_from(list(nx.isolates(tf_tf_GM)))
    tf_tf_GNM.remove_nodes_from(list(nx.isolates(tf_tf_GNM)))
   
    mcc = nx.number_connected_components(tf_tf_GM)
    nmcc = nx.number_connected_components(tf_tf_GNM)
   
    net_topology.loc['Number of nodes'] = [ nodes_m, nodes_nm]
    net_topology.loc['Number of edges'] = [n_edges_m, n_edges_nm]
    net_topology.loc['Number of singletons'] = [n_singltn_m, n_singltn_nm]
    net_topology.loc['Number of connected components'] = [mcc, nmcc]
    net_topology.loc['Size of largest component'] = [lcc_m, lcc_nm]
    net_topology.loc['Average degree'] = [avg_deg_m, avg_deg_nm]
    net_topology.loc['Largest degree'] = [l_deg_m, l_deg_nm]
    net_topology.loc['Clustering coefficient'] = [avg_cc_m, avg_cc_nm]
    print("TF-TF network topology: \n", net_topology)
#%% TFTG coregulatory network analysis and statistics
r2_threshold = 0.3
goodGenes = (r2.values >= r2_threshold)
filtered_meta = lasso_meta.loc[goodGenes, :] # filtering genes with better r2
filtered_nmeta = lasso_nmeta.loc[goodGenes, :] # filtering genes with better r2

target_tf_meta = pd.DataFrame(filtered_meta)
target_tf_nmeta = pd.DataFrame(filtered_nmeta)
cols = list(map(int, target_tf_meta.columns))

target_tf_meta.columns = cols
target_tf_nmeta.columns = cols

all_targets = np.union1d(target_tf_meta.index, target_tf_meta.columns)
adj_mat_meta = target_tf_meta.reindex(all_targets, columns = all_targets, fill_value = 0).T
adj_mat_nmeta = target_tf_nmeta.reindex(all_targets, columns = all_targets, fill_value = 0).T

pos_edge_m, neg_edge_m = (adj_mat_meta > 0).sum().sum(), (adj_mat_meta < 0).sum().sum()
pos_edge_nm, neg_edge_nm = (adj_mat_nmeta > 0).sum().sum(), (adj_mat_nmeta < 0).sum().sum()

GM = nx.convert_matrix.from_pandas_adjacency(adj_mat_meta, create_using=nx.DiGraph())
GNM = nx.convert_matrix.from_pandas_adjacency(adj_mat_nmeta, create_using=nx.DiGraph())

GM.edges(data = True)
GNM.edges(data = True)

print(GM)
print(GNM)

n_edges_m, n_edges_nm = nx.number_of_edges(GM), nx.number_of_edges(GNM)
n_singletons_m, n_singletons_nm = len(list(nx.isolates(GM))), len(list(nx.isolates(GNM)))

def get_in_out_degree_dist(G1, G2, targets, tfs):
    in_degrees_m = [G1.in_degree(n) for n in targets]
    in_degrees_nm = [G2.in_degree(n) for n in targets]
    out_degrees_m = [G1.out_degree(n) for n in tfs]
    out_degrees_nm = [G2.out_degree(n) for n in tfs]
    
    ##Removing zeors from the list for the TFs. For targets we have at least 1 TF that's regulating it.
    out_degrees_m = [i for i in out_degrees_m if i != 0]
    out_degrees_nm = [i for i in out_degrees_nm if i != 0]
    return in_degrees_m,in_degrees_nm, out_degrees_m,out_degrees_nm
   

#Getting Tfs and targets
targets = target_tf_meta.index
tfs = target_tf_meta.columns

in_meta, in_nmeta, out_meta, out_nmeta = get_in_out_degree_dist(GM, GNM, targets, tfs)

#getting number of TFs and targets with at least 1 target or TF in it.
n_tf_meta, n_tf_nmeta = len(out_meta), len(out_nmeta)
n_tg_meta, n_tg_nmeta = len(in_meta), len(in_nmeta)

#Getting minimum or maximum in and out degree
max_out_m, max_out_nm = max(out_meta), max(out_nmeta)
max_in_m, max_in_nm = max(in_meta), max(in_nmeta)

#Getting average and median in and out degrees.
avg_in_meta, avg_in_nmeta = np.mean(in_meta), np.mean(in_nmeta)
avg_out_meta, avg_out_nmeta = np.mean(out_meta), np.mean(out_nmeta)
med_in_meta, med_in_nmeta = np.median(in_meta), np.median(in_nmeta)
med_out_meta, med_out_nmeta = np.median(out_meta), np.median(out_nmeta)

# #removing the singletons
GM.remove_nodes_from(list(nx.isolates(GM)))
GNM.remove_nodes_from(list(nx.isolates(GNM)))

##calculating nodes after removing the sinfletons
nodes_m, nodes_nm = GM.number_of_nodes(), GNM.number_of_nodes()
#average degree
avg_deg_m, avg_deg_nm = n_edges_m /nodes_m ,  n_edges_nm / nodes_nm

#largest degree
l_deg_m = sorted(GM.degree, key=lambda x: x[1], reverse=True)[0][1]
l_deg_nm = sorted(GNM.degree, key=lambda x: x[1], reverse=True)[0][1]
#Clustering coefficient
avg_cc_m,  avg_cc_nm= nx.average_clustering(GM), nx.average_clustering(GNM)

#getting average shortest path
avg_spl_m, avg_spl_nm = nx.average_shortest_path_length(GM), nx.average_shortest_path_length(GNM)

#Converting the graph into undriected graph because for directed graph some functions are unavailable.
HGM, HGNM = GM.to_undirected(), GNM.to_undirected()

#getting largest connected components
gcc_m = sorted(nx.connected_components(HGM), key=len, reverse=True)
lcc_m = len(HGM.subgraph(gcc_m[0]))
gcc_nm = sorted(nx.connected_components(HGNM), key=len, reverse=True)
lcc_nm = len(HGNM.subgraph(gcc_nm[0]))

diameter_m, diameter_nm = nx.diameter(HGM), nx.diameter(HGNM)

#Clustering coefficient
avg_cc_m_ud,  avg_cc_nm_ud= nx.average_clustering(HGM), nx.average_clustering(HGNM)

#getting average shortest path
avg_spl_m_ud, avg_spl_nm_ud = nx.average_shortest_path_length(HGM),\
                                nx.average_shortest_path_length(HGNM)

#average degree
nodes_m_ud, nodes_nm_ud = HGM.number_of_nodes(), HGNM.number_of_nodes()
avg_deg_m_ud, avg_deg_nm_ud = 2*HGM.number_of_edges() /nodes_m_ud ,\
                                2*HGNM.number_of_edges() / nodes_nm_ud

print('Network stats \t\t\t meta \t non-meta')

print('# of nodes \t\t\t\t{} \t{}'.format(nodes_m, nodes_nm))
print('# of edges \t\t\t\t{} \t{}'.format(n_edges_m, n_edges_nm))


print('# of TFs \t\t\t\t{} \t{}'.format(n_tf_meta, n_tf_nmeta))
print('# of Targets \t\t\t\t{} \t{}'.format(n_tg_meta, n_tg_nmeta))



print('+ve edge \t\t\t\t{} \t {}'.format(pos_edge_m, pos_edge_nm))
print('-ve edge \t\t\t\t{} \t {}'.format(neg_edge_m, neg_edge_nm))


print('max out degree(TF) \t\t{} \t\t {}'.format(max_out_m, max_out_nm))
print('max in degree(TG) \t\t{} \t\t {}'.format(max_in_m, max_in_nm))
print('mean out degree(TF) \t\t{:.2f} \t {:.2f}'.format(avg_out_meta, avg_out_nmeta))
print('mean in degree(TG)\t\t{:.2f} \t {:.2f}'.format(avg_in_meta, avg_in_nmeta ))
print('median out degree(TF) \t\t{:.2f} \t {:.2f}'.format(med_out_meta, med_out_nmeta))
print('median in degree(TG)\t\t{:.2f} \t {:.2f}'.format(med_in_meta, med_in_nmeta))



print('singletons \t\t\t\t{} \t\t {}'.format(n_singletons_m, n_singletons_nm))
print('mean degree \t\t\t\t{:.2f} \t {:.2f}'.format(avg_deg_m, avg_deg_nm))

print('largest degree \t\t\t{} \t\t {}'.format(l_deg_m, l_deg_nm))

print('Average CC \t\t\t\t{:.2f} \t {:.2f}'.format(avg_cc_m, avg_cc_nm))
print('Average Shortest path \t{:.2f} \t {:.2f}'.format(avg_spl_m, avg_spl_nm))

print('mean degree UD \t\t\t\t{:.2f} \t {:.2f}'.format(avg_deg_m_ud, avg_deg_nm_ud))
print('Average CC UD \t\t\t\t{:.2f} \t {:.2f}'.format(avg_cc_m_ud,  avg_cc_nm_ud))
print('Average Shortest path UD \t{:.2f} \t {:.2f}'.format(avg_spl_m_ud, avg_spl_nm_ud))

print('Largest connected component UD {} \t {}'.format(lcc_m, lcc_nm))
print('Diameter UD \t\t\t\t{} \t\t {}'.format(diameter_m, diameter_nm))

#%% TF-Target network creation and node difference calculation
'''Creating TF-Target network'''
r2_threshold = 0.3
goodGenes = (r2.values >= r2_threshold)
filtered_meta = lasso_meta.loc[goodGenes, :] # filtering genes with better r2
filtered_nmeta = lasso_nmeta.loc[goodGenes, :] # filtering genes with better r2

coef_th = 0.0
target_tf_meta = pd.DataFrame(np.abs(filtered_meta) > coef_th, dtype = np.int) #setting coef >= coef_th as 1 else 0
target_tf_nmeta = pd.DataFrame(np.abs(filtered_nmeta) > coef_th, dtype = np.int)#setting coef >= coef_th as 1 else 0

cols = list(map(int, target_tf_meta.columns))

target_tf_meta.columns = cols
target_tf_nmeta.columns = cols


all_targets = np.union1d(target_tf_meta.index, target_tf_meta.columns)

adj_mat_meta = target_tf_meta.reindex(all_targets, columns = all_targets, fill_value = 0).T
adj_mat_nmeta = target_tf_nmeta.reindex(all_targets, columns = all_targets, fill_value = 0).T


GM = nx.convert_matrix.from_pandas_adjacency(adj_mat_meta)
GNM = nx.convert_matrix.from_pandas_adjacency(adj_mat_nmeta)


df_attr_m = pd.DataFrame(columns = ['nodes', 'degree'], data =  GM.degree())
df_attr_nm = pd.DataFrame(columns = ['nodes', 'degree'], data =  GNM.degree())

df_attr_m['node_type'] = ['tf' if x in target_tf_meta.columns else 'target' for x in df_attr_m['nodes']]
df_attr_nm['node_type'] = ['tf' if x in target_tf_nmeta.columns else 'target' for x in df_attr_nm['nodes']]

'''Uncomment these if you want to attribute files'''
# df_attr_m.to_csv(root_path + '/TFTargetNetwork/all_patient_tftg_net_meta_attr_coef_th_{0}_r2_{1}_alpha_{2}.csv'.\
#                   format(coef_th, r2_threshold, alpha_val))
# df_attr_nm.to_csv(root_path + '/TFTargetNetwork/all_patient_tftg_net_nmeta_attr_coef_th_{0}_r2_{1}_alpha_{2}.csv'.\
#                   format(coef_th, r2_threshold, alpha_val))
'''
Calculating the Node Difference usinf Li Lu's formula'
'''
all_nodes = sorted(GM.nodes)
euclidean_dists = {}
for p in all_nodes:
    dist_sum = 0
    for q in all_nodes:
        pij = GM.number_of_edges(p, q)
        qij = GNM.number_of_edges(p, q)
        dist_sum += np.abs(pij - qij)
    Di = np.sqrt(dist_sum)
    euclidean_dists[p] = round(Di, 3)
nx.set_node_attributes(GM, euclidean_dists, 'differences')
nx.set_node_attributes(GNM, euclidean_dists, 'differences')
# GM.remove_nodes_from(list(nx.isolates(GM)))
# GNM.remove_nodes_from(list(nx.isolates(GNM)))
print(nx.info(GM))
print(nx.info(GNM))

save_m = root_path + '/all_patient_tg_tf_net_meta_coef_th{0}_r2_{1}_alpha_{2}.gml'.\
        format(coef_th, r2_threshold, alpha_val)
save_nm = root_path + '/all_patient_tg_tf_net_nmeta_coef_th{0}_r2_{1}_alpha_{2}.gml'.\
        format(coef_th, r2_threshold, alpha_val)
# nx.write_gml(GM, save_m)
# nx.write_gml(GNM, save_nm)


#%% Read and save Node Stats from cytoscape files
##Got these files from cytoscape after importing the all_patient_tg_tf_net_meta_coef_th...gml files

fmeta = './TFTargetNetwork/all_patient_tg_tf_net_meta_coef_th0.0_r2_0.3_alpha_0.1node.csv'
fnmeta = './TFTargetNetwork/all_patient_tg_tf_net_nmeta_coef_th0.0_r2_0.3_alpha_0.1node.csv'
col_meta = ['deg_m', 'diff_m', 'HGNC_m', 'label_m', 'node_type_m']
col_nmeta = ['deg_nm', 'diff_nm', 'HGNC_nm', 'label_nm', 'node_type_nm']
mstats = pd.read_csv(fmeta, index_col = 0)
nmstats = pd.read_csv(fnmeta, index_col=0)
joined_stats = pd.concat([mstats, nmstats], axis = 1)
joined_stats.columns = col_meta + col_nmeta
top_tfs = joined_stats[joined_stats['node_type_m'] == 'tf'].sort_values('diff_m', ascending = False)
top_targets = joined_stats[joined_stats['node_type_m'] == 'target'].sort_values('diff_m', ascending = False)

#Uncomment these lines if you want to save these two files

# top_tfs.to_csv('top_tfs_based_on_diff.csv')
# top_targets.to_csv('top_targets_based_on_diff.csv')
#%%Getting Subgraphs/Subnetworks for targets and TFs for targets
r2_threshold = 0.3
goodGenes = (r2.values >= r2_threshold)
filtered_meta = lasso_meta.loc[goodGenes, :] # filtering genes with better r2
filtered_nmeta = lasso_nmeta.loc[goodGenes, :] # filtering genes with better r2

coef_th = 0.0
target_tf_meta = pd.DataFrame(filtered_meta, dtype = np.float) #setting coef >= coef_th as 1 else 0
target_tf_nmeta = pd.DataFrame(filtered_nmeta, dtype = np.float)#setting coef >= coef_th as 1 else 0

cols = list(map(int, target_tf_meta.columns))

target_tf_meta.columns = cols
target_tf_nmeta.columns = cols

all_targets = np.union1d(target_tf_meta.index, target_tf_meta.columns)

adj_mat_meta = target_tf_meta.reindex(all_targets, columns = all_targets, fill_value = 0).T
adj_mat_nmeta = target_tf_nmeta.reindex(all_targets, columns = all_targets, fill_value = 0).T

GM = nx.convert_matrix.from_pandas_adjacency(adj_mat_meta)
GNM = nx.convert_matrix.from_pandas_adjacency(adj_mat_nmeta)

GM.edges(data = True)
GNM.edges(data = True)


mstats = pd.read_csv('./Dataset/all_patient_tg_tf_net_meta_coef_th0.0_r2_0.3_alpha_0.1node.csv', \
                     index_col = 4)

top_tfs = pd.read_csv('./Dataset/top_tfs_based_on_diff.csv', index_col=0)
top_targets = pd.read_csv('./Dataset/top_targets_based_on_diff.csv', index_col=0)

top_targets.set_index('label_m', inplace = True)


all_nodes = top_targets[:10]
node_list1 = list(GM.nodes)
node_list2 = list(GNM.nodes)
node_list1.extend(node_list2)

for node in all_nodes.index:
    neighbor_m = [n for n in GM.neighbors(node)]
    HM = GM.subgraph(neighbor_m+[node])
    for e in HM.edges(data=True):
        HM.edges[e[0], e[1]]['sign'] = int(e[2]['weight'] > 0)
    neighbor_nm = [n for n in GNM.neighbors(node)]
    
    HNM = GNM.subgraph(neighbor_nm+[node])
    for e in HNM.edges(data=True):
        HNM.edges[e[0], e[1]]['sign'] = int(e[2]['weight'] > 0)
    node_name = top_targets.loc[node, 'HGNC_m']
    print(node_name)

    HM = nx.relabel_nodes(HM, lambda x: mstats.loc[x, 'HGNC'] if x != None else x)
    HNM = nx.relabel_nodes(HNM, lambda x: mstats.loc[x, 'HGNC'] if x != None else x)
    
    sub_m = './meta_sub_net_{0}.gml'.format(node_name)
    sub_nm = './nmeta_sub_net_{0}.gml'.format(node_name)
    #Uncomment these lines if you want to save these networks
    # nx.write_gml(HM, sub_m)
    # nx.write_gml(HNM, sub_nm)
    # nx.draw_networkx(HM)
    plt.show()