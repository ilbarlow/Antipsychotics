#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:28:32 2018

@author: ibarlow
"""

#run analysis using only the liquids data
    #NOTES - #need to impute nans before Zscoring and normalisation


#load data
    
from tkinter import Tk, filedialog
import pandas as pd
import os
import TierPsyInput as TP


#now call these functions
directoryA, fileDirA, featuresA = TP.TierPsyInput('new', 'Agar')

#%%
#from here pull out info about each of the experiment's variables, such as drug, concentration and uniqueID
import numpy as np

#make a copy of featuresA
featuresA2 = featuresA.copy()

#pop out experiment list
exp_namesA={}
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
for rep in drugA:
    drugs.append(list(drugA[rep].values))
    concs.append(list(concA[rep].values))
flatten = lambda LoL: [item for sublist in LoL for item in sublist] #lambda function flattens list of lists (LoL)

uniqueDrugs = list(np.unique(flatten(drugs)))
uniqueConcs = list(np.unique(flatten(concs)))
del drugs, concs

#%%
#now for filtering of tracks and features

import numpy as np

to_excludeA={}
for rep in featuresA:
    to_excludeA[rep] = TP.FeatFilter(featuresA[rep])

#combined for all experiments to exclude
list_exclude = [y for v in to_excludeA.values() for y in v]
list_exclude = list(np.unique(list_exclude))

#now drop these features
for rep in featuresA:
    featuresA[rep].drop(list_exclude, axis = 1, inplace= True)
    featuresA[rep] = featuresA[rep].reset_index(drop = True)


#%% Z-score normalisation

featuresZ={}
for rep in featuresA:
    featuresZ[rep] = TP.z_score(featuresA[rep])

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
import numpy as np


#algorithm doesn't work with NaNs, so need to impute:
    #one inputation with mean, and another with median
featMatTotalNorm_mean = {}
featMatTotalNorm_med = {}
featMatTotal_mean = {}
for rep in featuresZ1:
    featMatTotalNorm_mean[rep] = featuresZ1[rep].fillna(featuresZ1[rep].mean(axis = 0))
    featMatTotalNorm_med[rep] = featuresZ1[rep].fillna(featuresZ1[rep].median(axis = 0))
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
for rep in featuresA2:
    featuresA2 [rep] ['drug'] = drugA[rep]
    featuresA2[rep] ['concentration'] = concA[rep]
    #featuresA2[rep]['exp'] =exp_namesA[rep]
    featuresA2[rep] ['date'] = dateA[rep]
    
#compare each compound to control data
controlMeans = {}
for rep in featuresA2:
    controlMeans[rep] = pd.DataFrame()
    for line in range(len(featuresA2[rep])):
        if featuresA2[rep]['drug'].iloc[line] == 'DMSO':
            DMSO = featuresA2[rep].iloc[line]
            controlMeans[rep] = controlMeans[rep].append(DMSO.to_frame().transpose())
            del DMSO
    controlMeans[rep] = controlMeans[rep].reset_index(drop=True)

#so can now compile a matrix of p_values comparing each drug to control means
    #need to loop through the different drugs and basically just compare all the features to controlMeans
pVals = {}
feats =list(featuresA2[rep].columns)
feats = feats[0:-2]
for rep in featuresA2:
    pVals[rep] = pd.DataFrame()
    for drug in uniqueDrugs:
        if drug == 'DMSO':
            continue
        else:        
            currentInds = pd.DataFrame()
            for line in featuresA2[rep].iterrows():
                if line[1]['drug'] ==drug:
                    current = line[1].to_frame().transpose()
                    currentInds = currentInds.append(current)
                    del current
            
            currentInds = currentInds.reset_index(drop=True)
            
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
                    for feature in currentInds.columns[0:-4]:
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
    plt.figure()
    sns.swarmplot(x='drug', y=feature, data=features_df, \
                      hue = 'concentration', palette = cmap)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5) ,ncol = 1, frameon= True)    
    plt.xticks(rotation = 45)
    plt.savefig(os.path.join (directoryA[0:-7], 'Figures', rep1 + feature + file_type),\
                 bbox_inches="tight", dpi = 200) 

for rep in featuresA2:
    for feat in range(0,10):
        swarms (rep, sig_feats[rep][feat][0], featuresA2[rep], directoryA, '.tif')


#so looks like can pull out differences between DMSO and chloropromazine and clozapine. Data is messier probably
        #because only one repeat

#%% move on to tSNE and clustering - may not work very well as not much data here

import tSNE_custom as SNE

#define featuresZ2
#first z-score the features
featuresZ2 = {}
featuresA3 = featuresA2.copy()
featMatAll = pd.DataFrame()
for rep in featuresA3:
    #reset index
    featuresA3[rep] = featuresA3[rep].reset_index(drop=True)
    #impute nans
    featuresA3[rep] = featuresA3[rep].fillna(featuresA3[rep].mean(axis=0))
    #zscore
    np.seterr(divide='ignore', invalid='ignore')
    featuresZ2[rep] = pd.DataFrame(stats.zscore(featuresA3[rep].iloc[:,:-3], axis=0))
    
    featuresZ2[rep].columns = featuresA3[rep].iloc[:,:-3].columns
    featuresZ2[rep] = TP.FeatRemove(featuresZ2[rep], list_exclude) #remove nan features
    
    #add in drugs, doses and dates
    featuresZ2[rep]= pd.concat([featuresZ2[rep], featuresA3[rep].iloc[:,-3:]],  axis = 1)
    
    #concat into a big features dataframe
    featMatAll = featMatAll.append(featuresZ2[rep])

#reset index
featMatAll = featMatAll.reset_index(drop = True)

tSNE_1 = {}
times = {}
testing = list(np.arange (1,102,20))

#know that perplexity =20 is best to just use that

tSNE_all, times_all = SNE.tSNE_custom(featMatAll, testing)

SNE.sne_plot(tSNE_all, testing, [], uniqueConcs)

#not enough data to do the clustering but looks like there is some resolution between drugs
