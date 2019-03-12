#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:33:10 2018

@author: ibarlow
"""
""" analysis antipsychotics data using new featuers and egg data"""

#%%

import TierPsyInput as TP
import numpy as np
import pandas as pd

directoryA, fileDirA, featuresA = TP.TierPsyInput('new', 'Liquid')

#from here pull out info about each of the experiment's variables, such as drug, concentration and uniqueID

    #first, remove experiment list
exp_namesA={}
featuresA2 = featuresA.copy()
for rep in featuresA:
    exp_namesA[rep] = featuresA[rep].pop('exp')

drugA = {}
concA = {}
dateA = {}
uniqueIDA = {}

for rep in exp_namesA:
    drugA[rep], concA[rep], dateA[rep], uniqueIDA[rep] = TP.extractVars(exp_namesA[rep])


#make lists of unqiue drugs and concs
drugs = []
concs = []
dates =[]
for rep in drugA:
    drugs.append(list(drugA[rep].values))
    concs.append(list(concA[rep].values))
    dates.append(list(dateA[rep].values))
flatten = lambda LoL: [item for sublist in LoL for item in sublist] #lambda function flattens list of lists (LoL)

uniqueDrugs = list(np.unique(flatten(drugs)))
uniqueConcs = list(np.unique(flatten(concs)))
uniqueDates = list(np.unique(flatten(dates)))
del drugs, concs, dates

#%% import eggs data
import os


#fileid for eggs data
fid_eggs = os.path.join (directoryA[:-7], 'egg_all.csv')
#load data
eggs_df = pd.read_csv(fid_eggs)

#split the data by experiment
eggs={}
reps = list(fileDirA.keys())

#make dictionary to match rep and date
rep_match = {}
for rep in dateA:
    rep_match[list(np.unique(dateA[rep].values))[0]] = rep

#and then compile dictionary using this   
for date in uniqueDates:
    eggs[rep_match[date]] = eggs_df[eggs_df['date'] ==int(date)]
    eggs[rep_match[date]] = eggs[rep_match[date]].reset_index(drop=True)

del rep_match, reps
#problem is that there is not egg data for every plate... so need to match up data accordingly

#add on descriptors to A2 dataframe
featuresA3 = {}
for rep in featuresA2:
    featuresA3[rep] = pd.concat([featuresA2[rep], dateA[rep], drugA[rep], concA[rep], uniqueIDA[rep]], axis =1)

#now to match them up
eggs2 = eggs.copy()
eggs_df2 = {}
for rep in featuresA3:
    eggs_df2[rep] = pd.DataFrame()
    for step in range(0, featuresA3[rep].shape[0]):
        line = featuresA3[rep].iloc[step,:]
        temp = eggs[rep][eggs[rep]['uniqueID'] == line['uniqueID']]
        if temp.shape[0]>0:
            temp2 = temp[temp['drug'] == line['drug']]
            temp2 = temp2[temp2['concentration'] == line['concentration']]
            if temp2.shape[0] == 0:
                temp3 = line.copy()
                temp3['total'] = float('nan')
            else:
                if temp2.shape[0]>1:
                    temp3 = temp2.iloc[0,:] #this is so that only place the data for one if there are duplicates
                    eggs[rep] = eggs[rep].drop(temp3.to_frame().transpose().index)
                else:
                    temp3 = temp2
            del temp2
        else:
            #put nan in place
            temp3 = line.copy()
            temp3['total'] = float('nan')
            
            
        line = line.to_frame().transpose()   
        line['eggs'] = float(temp3['total'])
        
        eggs_df2[rep] = eggs_df2[rep].append(line)
        del line, temp, temp3

#Call this new dataframe featuresEA_1
featuresEA_1 = eggs_df2.copy()
    
del eggs2,featuresA3

#need to pop out the experiment descriptors again as may be wrong
drugA2 = {}
concA2 ={}
dateA2={}
uniqueIDA2 = {}

for rep in featuresEA_1:
    drugA2[rep] = featuresEA_1[rep].pop('drug')
    concA2[rep] = featuresEA_1[rep].pop('concentration')
    dateA2[rep] = featuresEA_1[rep].pop('date')
    uniqueIDA2[rep] = featuresEA_1[rep].pop('uniqueID')
    
#%% Z-score normalisation

featuresZ={}
for rep in featuresA:
    featuresZ[rep] = TP.z_score(featuresEA_1[rep])

to_excludeA={}
for rep in featuresZ:
    to_excludeA[rep] = TP.FeatFilter(featuresZ[rep])
    
#combined for all experiments to exclude
list_exclude = [y for v in to_excludeA.values() for y in v]
list_exclude = list(np.unique(list_exclude))

#remove these features from the features dataframes
featuresZ1 = {}
for rep in featuresZ:
    featuresZ1[rep] = TP.FeatRemove(featuresZ[rep], list_exclude)

#%% move on to PCA

#now to make covariance matrix for PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
import PCA_analysis as PC


#algorithm doesn't work with NaNs, so need to impute:
    #one inputation with mean, and another with median
featMatTotalNorm_mean = {}
featMatTotalNorm_med = {}
featMatTotal_mean = {}
for rep in featuresZ1:
    featMatTotalNorm_mean[rep] = featuresZ1[rep].fillna(featuresZ1[rep].mean(axis = 0), inplace=True)
    featMatTotalNorm_med[rep] = featuresZ1[rep].fillna(featuresZ1[rep].median(axis = 0), inplace = True)
    featMatTotal_mean [rep]= featuresA[rep].fillna(featuresA[rep].mean(axis=0))
    
#fit and transform data onto standard scale - this means that Z-score normalising was redundant
X_std1={}
#X_std2={}
exp_names = {}
cov_mat={}
for rep in featMatTotalNorm_mean:
    X_std1[rep] = StandardScaler().fit_transform(featMatTotalNorm_mean[rep])
    #X_std2[rep] = StandardScaler().fit_transform(features2[rep].iloc[:,4:-2]) #don't include the recording info in the PCA

    cov_mat[rep] = np.cov(X_std1[rep].T)
    #cov_mat2[rep] = np.cov(X_std2[rep].T)


eig_vecs1={}
eig_vals1 = {}
eig_pairs1 = {}
PC_pairs1={}
PC_df1 = {}
cut_off1={}

for rep in X_std1:
    eig_vecs1[rep], eig_vals1[rep], eig_pairs1[rep], PC_pairs1[rep],\
    PC_df1[rep], cut_off1[rep] = PC.pca(X_std1[rep], rep, directoryA, '.tif')
    

PC_conts1 = {}
PC_feats1 = {}
PC_top1={}
x1 ={}
for rep in eig_pairs1:
    PC_conts1[rep], PC_feats1[rep], \
    PC_top1[rep], x1[rep] = PC.PC_feats(eig_pairs1[rep], cut_off1[rep], featuresZ[rep])

#now make biplots for all the reps 
for rep in PC_top1:
    PC.biplot(PC_top1[rep], PC_feats1[rep],1,2, 1, directoryA, rep, '.tif', uniqueDrugs)
    
#%% now to transform into feature space
    #concanenate the eigen_vector matrix across the top 80 eigenvalues

matrix_w1 = {}
Y1 = {}
PC_df2 = {}
for rep in featuresZ1:
    matrix_w1[rep], Y1[rep], PC_df2[rep] = PC.feature_space(featuresZ1[rep], eig_pairs1[rep],\
            X_std1[rep], cut_off1[rep], x1[rep], drugA[rep], concA[rep], dateA[rep])

#set palette for plots
sns.palplot(sns.choose_colorbrewer_palette(data_type = 'q'))
#now make the plots   
for rep in PC_df2:
    for i in [1,10,100,200]:
        PC.PC12_plots(PC_df2[rep], i, rep, directoryA, 'tif')

#now can make dataframe containing means and column names to plot trajectories through PC space
PC_means1={}
for rep in PC_df2:
    PC_means1[rep] = PC.PC_av(PC_df2[rep], x1[rep])
    

sns.set_style('whitegrid')
for rep in PC_means1:
    PC.PC_traj(PC_means1[rep], rep,directoryA, 'tif')

#%% now to do the stats on the experiments
    
from scipy import stats

#for this it is usful to append the conditions onto the dataframe
for rep in featuresEA_1:
    featuresEA_1 [rep] ['drug'] = drugA2[rep]
    featuresEA_1[rep] ['concentration'] = concA2[rep]
    #featuresEA_1[rep]['exp'] =exp_namesA[rep]
    featuresEA_1[rep] ['date'] = dateA2[rep]
    
#compare each compound to control data
controlMeans = {}
for rep in featuresEA_1:
    controlMeans[rep] = pd.DataFrame()
    for line in range(len(featuresA2[rep])):
        if featuresEA_1[rep]['drug'].iloc[line] == 'DMSO':
            DMSO = featuresEA_1[rep].iloc[line]
            controlMeans[rep] = controlMeans[rep].append(DMSO.to_frame().transpose())
            del DMSO
    controlMeans[rep].reset_index(drop=True)

#so can now compile a matrix of p_values comparing each drug to control means
    #need to loop through the different drugs and basically just compare all the features to controlMeans
pVals = {}
feats =list(featuresEA_1[rep].columns)
feats = feats[0:-1]
for rep in featuresEA_1:
    pVals[rep] = pd.DataFrame()
    for drug in uniqueDrugs:
        if drug == 'DMSO':
            continue
        else:        
            currentInds = pd.DataFrame()
            for line in featuresEA_1[rep].iterrows():
                if line[1]['drug'] ==drug:
                    current = line[1].to_frame().transpose()
                    currentInds = currentInds.append(current)
                    del current
            
            currentInds.reset_index(drop=True)
            
            #test if dataframe is empty to continue to next drug
            if currentInds.empty:
                continue
            else:
                #separate the concentrations
                conc= np.unique(currentInds['concentration'])
                for dose in conc:
                    test =[]
                    to_test = currentInds['concentration'] ==dose
                    testing = currentInds[to_test]
                    for feature in currentInds.columns[0:-3]:
                        test.append(stats.ttest_ind(testing[feature], controlMeans[rep][feature]))
       
                    ps = [(test[i][1]) for i in range(len(test))] #make into a list
                    ps.append(drug)
                    ps.append(dose)
        
                    temp = pd.DataFrame(ps).transpose()
                    pVals[rep] = pVals[rep].append(temp)
                    del temp, to_test, testing
            del currentInds

    #add in features
    pVals[rep].columns = feats
    pVals[rep] = pVals[rep].reset_index (drop=True)   

#import module for multiple comparison correction
import statsmodels.stats.multitest as smm

#now correct for multiple comparisons
bh_p={}
top_feats = {}
post_exclude = {}
sig_feats = {}
for rep in pVals:
    bh_p [rep] = pd.DataFrame()
    for cond in range(pVals[rep].shape[0]):
        reg, corrP, t,s =  smm.multipletests(pVals[rep].values[cond, :-2],\
                                             alpha=0.05, method = 'fdr_bh', \
                                             is_sorted = False, returnsorted = False)
        corrP = list(corrP)
        corrP.append(pVals[rep]['drug'].iloc[cond])
        corrP.append(pVals[rep]['concentration'].iloc[cond])
        corrP = pd.DataFrame(corrP).transpose()
        bh_p [rep]= bh_p[rep].append(corrP)
        del corrP
        #add in the feature names
    bh_p[rep].columns = feats
    bh_p [rep]= bh_p[rep].reset_index(drop = True)

    #now can filter out the features that have no significant p-values
    top_feats[rep]= bh_p[rep].values[:,0:-2] <=0.05
    post_exclude [rep]= []
    sig_feats[rep] = []
    for feat in range(bh_p[rep].shape[1]-2):
        if np.sum(top_feats[rep][:,feat]) ==0:
            post_exclude[rep].append(bh_p[rep].columns[feat])
        else:
            sig_feats[rep].append((bh_p[rep].columns [feat], np.sum(top_feats[rep][:,feat])))
    
    #sort by most to least
    sig_feats[rep].sort(key =lambda tup:tup[1])
    sig_feats[rep].reverse()


#make some violin plots of the significant features
cmap = sns.choose_colorbrewer_palette(data_type = 'q')
def swarms (rep1, feature, features_df, directory, file_type):
    """Makes swarm plot of features
    Input:
        rep1 - name of experiment
        
        feature - feature to be plotted
        
        features_df - dataframe of features
        
        directory - folder into which figure wil be saved
        
        file_type - image type to save (eg .tif, .svg)
    
    Output:
        swarm plot - 
    """
    
    sns.swarmplot(x='drug', y=feature, data=features_df, \
                      hue = 'concentration', palette = cmap)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5) ,ncol = 1, frameon= True)    
    plt.xticks(rotation = 45)
    plt.savefig(os.path.join (directoryA[0:-7], 'Figures', rep1 + feature + file_type),\
                 bbox_inches="tight", dpi = 200) 

for rep in featuresA2:
    for feat in range(0,10):
        swarms (rep, sig_feats[rep][feat][0], featuresA2[rep], directoryA, '.tif')
        
#%% combine all experiments
#use the full dataset (ie including all tracks for all doses and drugs)
    #internally z-score first and then concatenate, and then run t-sne on entire dataset
    
    #test different perplexities, but keep the n_iter at 1000
#prepare the dataframes

featuresZ2 = {}
featMatAll = pd.DataFrame()
for rep in featMatTotalNorm_mean:
    featuresZ2 [rep] = pd.concat([featMatTotalNorm_mean[rep], drugA2[rep], concA2[rep], dateA2[rep]], axis =1)
    featMatAll = featMatAll.append(featuresZ2[rep])

featMatAll = featMatAll.reset_index(drop = True)
featMatAll2 = featMatAll.copy()

drug_all = featMatAll2.pop ('drug')
conc_all = featMatAll2.pop('concentration')
date_all = featMatAll2.pop ('date')

#%% tSNE on the new features

import tSNE_custom as SNE

#define featuresZ2
#first z-score the features
featuresZ2 = {}
featuresA3 = featuresEA_1.copy()
featMatAll = pd.DataFrame()
for rep in featuresA3:
    #reset index
    featuresA3[rep] = featuresA3[rep].reset_index(drop=True)
    #impute nans
    featuresA3[rep] = featuresA3[rep].fillna(featuresA3[rep].mean(axis=0), inplace=True)
    #zscore
    np.seterr(divide='ignore', invalid='ignore') #to prevent from erroring
    featuresZ2[rep] = pd.DataFrame(stats.zscore(featuresA3[rep].iloc[:,:-3], axis=0))
    featuresZ2[rep].columns = featuresA3[rep].iloc[:,:-3].columns
    #add in drugs, doses and dates
    featuresZ2[rep]= pd.concat([featuresZ2[rep], featuresA3[rep].iloc[:,-3:]],  axis = 1)
    featuresZ2[rep] = TP.FeatRemove(featuresZ2[rep], list_exclude)
    
    #concat into a big features dataframe
    featMatAll = featMatAll.append(featuresZ2[rep])

#reset index
featMatAll = featMatAll.reset_index(drop = True)


tSNE_1 = {}
times = {}
testing = list(np.arange (1,102,20))

#know that perplexity =20 is best to just use that

for rep in featuresZ2:
    tSNE_1[rep], times[rep] = SNE.tSNE_custom(featuresZ2[rep], 20)

tSNE_all, times_all = SNE.tSNE_custom(featMatAll, 20)

SNE.sne_plot(tSNE_1[rep],20, [], uniqueConcs)

SNE.sne_plot(tSNE_all, 20, [], uniqueConcs)

#%% clustering

#just try using Dcluster package
from scipy import spatial
import Dcluster as dcl
import os
from kneed import KneeLocator
import density_plots as dp

to_test =20
temp1 = spatial.distance.pdist(tSNE_all[to_test].iloc[:,0:2])
temp1 = spatial.distance.squareform(temp1)

#this make squareform distance matrix so assume NDx == NDy

#save temp1 as a .txt file
fileid1 = os.path.join(directoryA[0:-7], 'new_feat_pdist_eggs.txt')
np.savetxt(fileid1, temp1, delimiter = '\t', fmt = '%f')

#also save the file identifiers
fileid2 = os.path.join(directoryA[0:-7], 'new_ids_eggs.csv')
tSNE_all[to_test].to_csv(fileid2)


#load file
dist1 = np.genfromtxt(fileid1, delimiter = '\t')
ids = pd.read_csv(fileid2, delimiter = ',').drop(['Unnamed: 0'],axis=1)

#vectorise dist values
xxdist1 = [(dist1[t2].T) for t2 in range(0,len(dist1))]
xxdist1 = np.concatenate(xxdist1[:])
ND1 = dist1.shape[0]
N1 = xxdist1.shape[0]

#can use this squareform pdist matrix in dcluster - will be alot faster and save on memory as well
(XY1, S1) = dcl.mds(dist1)

#define rho-delta values
(rho1, delta1, ordrho1,dc1, nneigh1) = dcl.rhodelta(dist1, xxdist1, ND1, N1)

sns.set_style('whitegrid')
f,ax, rho_d = dp.plot1(rho1, delta1)
f.savefig(os.path.join(directoryA[0:-7], 'Figures', 'decision_graph_eggs.png'), dpi =200)

#based on these plots set rhomin and delta min
rhomin1 = 2
deltamin1 = 5

ax, clusters1 = dp.density_calc (dist1, XY1, ND1, rho1, delta1,ordrho1,dc1,nneigh1,rhomin1, deltamin1, directoryA[0:-7])
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'decision_graph2_eggs.png'), dpi =200)


dc_df = pd.DataFrame (data = XY1, columns = ['x', 'y'])
dc_df['cluster'] = clusters1[:,1]
dc_df['cluster_halo'] = clusters1[:,2]
dc_df['drug'] =ids['drug']
dc_df['concentration'] = ids['concentration']
N_clusters = int(max(clusters1[:,1]))

#make the plots
sns.set_palette(sns.color_palette("tab20", N_clusters))
sns.lmplot(x= 'x', y='y', data=dc_df, hue = 'cluster', fit_reg = False, legend=False)
plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
plt.tight_layout(rect=[0,0,0.75,1])
plt.axis('equal')
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'clusters__only_new_eggs.png'), dpi = 200)

sns.set_palette(sns.color_palette("tab20", len(uniqueDrugs)))
sns.lmplot(x= 'x', y='y', data=dc_df, hue = 'drug', fit_reg = False, legend=False)
plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
plt.tight_layout(rect=[0,0,0.75,1])
plt.axis('equal')
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'drugs_clusters_new_eggs.png'), dpi = 200, bbox_inches = 'tight')

#%% next part is to show which drugs make up which cluster
    #for each cluster find out percentage of each drug

clust_cont = {}
clust_cont['total'] = pd.Series()
for drug in uniqueDrugs:
    clust_cont[drug] = pd.DataFrame()
    temp1 = dc_df[dc_df['drug'] == drug]
    for conc in uniqueConcs:
        if temp1[temp1['concentration']==conc].empty ==False:
            temp2 =pd.Series()
            temp2['concentration'] = conc
            temp2['drug'] = conc
            for cluster in range(1, N_clusters +1):
                
                temp2[str(cluster)] = temp1[temp1['concentration']==conc]\
                [temp1['cluster'] == cluster].shape[0]
                
                clust_cont['total'][str(cluster)] = dc_df[dc_df['cluster']==cluster].shape[0]
                
                
            clust_cont[drug] = clust_cont[drug].append(temp2.to_frame().transpose())
            del temp2

        else:
            continue
    
    clust_cont[drug] = clust_cont[drug].reset_index(drop=True)
    del temp1
              
clust_cont['total'] = clust_cont['total'].to_frame().transpose()                


#now calculate percentage
clust_cont2 = {}
for drug in clust_cont:
    if drug == 'total':
        continue
    else:
        clust_cont2[drug] = pd.DataFrame()
        for conc in clust_cont[drug]['concentration']:
            temp1 = clust_cont[drug][clust_cont[drug]['concentration'] == conc]
            temp1 = temp1.reset_index(drop=True)
            temp2 = temp1/clust_cont['total']
            temp2['drug'] = drug  + ', ' + str(conc)
            temp2['concentration'] = conc
            clust_cont2[drug] = clust_cont2[drug].append(temp2)
            del temp1, temp2
        if clust_cont2[drug].shape[0]>1:
            temp3 = clust_cont[drug].sum(axis=0)/clust_cont['total']
            temp3['drug'] = drug
            temp3['concentration'] = 1000.00
            clust_cont2[drug] = clust_cont2[drug].append(temp3)
            del temp3
        else:
            continue      
    clust_cont2[drug] = clust_cont2[drug].reset_index(drop = True)

#concatenate
clust_all = pd.DataFrame()
for drug in clust_cont2:
    clust_all = clust_all.append(clust_cont2[drug])
clust_all = clust_all.reset_index(drop = True)

#clustermap
cg=sns.clustermap(data= clust_all.iloc[:,:-2], metric = 'euclidean', cmap = 'inferno')
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels(clust_all.iloc[cg.dendrogram_row.reordered_ind, -1]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)           
hm = cg.ax_heatmap.get_position()
cg.ax_heatmap.set_position([hm.x0, hm.y0, hm.width*0.75, hm.height])
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*0.75, col.height*0.75])
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'HC_clusters_new_Eggs.png'), dpi = 150)

    