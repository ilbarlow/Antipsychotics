#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:59:58 2018

@author: ibarlow
"""

""" Script to compare liquids and agar features to help decide whether useful
Steps to do:
    1. Import liquids data and do PCA, statistical comparisons and then run mRMR and classifier
    
    2. Import agar data and do the same to see if there are comparable differences
    
    FIGURES to make:
        PCA
        Swarms
        Clustergram?"""
        
#load data
 
import pandas as pd
import os
import TierPsyInput as TP


#now call these functions
    #select results folder
directoryL, fileDirL, featuresL = TP.TierPsyInput('new', 'Agar')

#%%
#from here pull out info about each of the experiment's variables, such as drug, concentration and uniqueID
import numpy as np

#make a copy of featuresA
featuresL2 = featuresL.copy()

#pop out experiment list
exp_namesL={}
for rep in featuresL:
    exp_namesL[rep] = featuresL[rep].pop('exp')

drugL = {}
concL = {}
dateL = {}
uniqueIDL = {}
for rep in exp_namesL:
    drugL[rep], concL[rep], dateL[rep], uniqueIDL[rep] = TP.extractVars(exp_namesL[rep])
    concL[rep] = concL[rep].fillna(0)
    
#make lists of unqiue drugs and concs
drugs = []
concs = []
for rep in drugL:
    drugs.append(list(drugL[rep].values))
    concs.append(list(concL[rep].values))
flatten = lambda LoL: [item for sublist in LoL for item in sublist] #lambda function flattens list of lists (LoL)

uniqueDrugs = list(np.unique(flatten(drugs)))
uniqueConcs = list(np.unique(flatten(concs)))
del drugs, concs

#%%
#now for filtering of tracks and features

import numpy as np

to_excludeL={}
for rep in featuresL:
    to_excludeL[rep] = TP.FeatFilter(featuresL[rep])

#combined for all experiments to exclude
list_exclude = [y for v in to_excludeL.values() for y in v]
list_exclude = list(np.unique(list_exclude))

#now drop these features
for rep in featuresL:
    featuresL[rep].drop(list_exclude, axis = 1, inplace= True)
    featuresL[rep] = featuresL[rep].reset_index(drop = True)


#%% Z-score normalisation

featuresZL={}
for rep in featuresL:
    featuresZL[rep] = TP.z_score(featuresL[rep])

to_excludeL={}
for rep in featuresZL:
    to_excludeL[rep] = TP.FeatFilter(featuresZL[rep])
    
#combined for all experiments to exclude
        #this time after Z-scoring
list_exclude = [y for v in to_excludeL.values() for y in v]
list_exclude = list(np.unique(list_exclude))

#remove these features from the features dataframes
featuresZL1 = {}
for rep in featuresZL:
    featuresZL1[rep] = TP.FeatRemove(featuresZL[rep], list_exclude)


#%%     
#data normalization
    #first impute the nans in the Z-scored data for each rep
#combine the Z-scored data as in the tSNE - this is better for when combining all the experiments
     
featMatTotalNormL_mean = {}
for rep in featuresZL1:
    featMatTotalNormL_mean[rep] = featuresZL1[rep].fillna(featuresZL1[rep].mean(axis = 0))
    
featuresZL2 = {}
featMatAllL = pd.DataFrame()
for rep in featMatTotalNormL_mean:
    featuresZL2 [rep] = pd.concat([featMatTotalNormL_mean[rep], drugL[rep], concL[rep], dateL[rep]], axis =1)
    featMatAllL = featMatAllL.append(featuresZL2[rep])

#reset index
featMatAllL = featMatAllL.reset_index(drop = True)
featMatAllL2 = featMatAllL.copy()

drug_allL = featMatAllL2.pop ('drug')
conc_allL = featMatAllL2.pop('concentration')
date_allL = featMatAllL2.pop ('date')

repsL = list(featuresZL2.keys())

#%% so now onto the PCA
    #use sklearn toolkit as more reliable, faster and more concise than my own code

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#make array of z-scored data
X = np.array(featMatAllL2.values)
#initialise PCA
pca = PCA()
X2= pca.fit_transform(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
thresh = cumvar <= 0.95 #set 95% variance threshold
cut_off = int(np.argwhere(thresh)[-1])

#make a plot
sns.set_style('whitegrid')
plt.plot(range(0, len(cumvar)), cumvar)
plt.plot([cut_off, cut_off], [0, 1], 'k')
plt.xlabel('Number of Principal Components')
plt.ylabel('variance explained')
plt.savefig(os.path.join(directoryL[:-7], 'Figures', 'liquidPCvar.png'), dpi =150)

#now put the 1:cut_off PCs into a dataframe
PCname = ['PC_%d' %(p+1) for p in range (0,cut_off+1)]
PC_df = pd.DataFrame(data= X2[:,:cut_off+1], columns = PCname)
PC_df['drug'] = drug_allL
PC_df['concentration'] = conc_allL
PC_df['date'] = date_allL

#components that explain the variance
    #make a dataframe ranking the features for each PC1 and also include the explained variance
PC_1feat = list(featMatAllL2.columns[np.argsort(pca.components_[0])])
PC_1vals = list(np.sort(pca.components_[0]))
PC_2feat = list(featMatAllL2.columns[np.argsort(pca.components_[1])])
PC_2vals = list(np.sort(pca.components_[1]))

PC_feat = []
PC_sum =[]
for PC in range(0, len(PCname)):
    PC_feat.append(list(featMatAllL2.columns[np.argsort(pca.components_[PC])]))
    PC_sum.append(list((pca.components_[PC])-\
                        np.mean((pca.components_[PC])))/np.std((pca.components_[PC])))
    
#dataframe containing standards cores of contibution of feature
PC_vals = pd.DataFrame(data= PC_sum, columns =  featMatAllL2.columns)

#okay so now can plot as biplot
plt.figure()
for i in range(0,2):
    plt.arrow(0,0, PC_vals.iloc[0,:][PC_feat[0][-1-i]], PC_vals.iloc[1,:][PC_feat[0][-1-i]],color= 'b')
    plt.arrow(0,0, PC_vals.iloc[0,:][PC_feat[1][-1-i]], PC_vals.iloc[1,:][PC_feat[1][-1-i]], color='r')
    plt.text(PC_vals.iloc[0,:][PC_feat[0][-1-i]] + 0.5, PC_vals.iloc[1,:][PC_feat[0][-1-i]] + 0.5, PC_feat[0][-1-i],\
             ha='center', va='center')
    plt.text(PC_vals.iloc[0,:][PC_feat[1][-1-i]]+0.5, PC_vals.iloc[1,:][PC_feat[1][-1-i]]+0.5,\
         PC_feat[1][-1-i], ha='center', va='center')

plt.xlim (-3, 3)
plt.ylim (-3,3)
plt.xlabel('PC_1')
plt.ylabel('PC_2')
plt.show()
plt.savefig(os.path.join(directoryL[:-7], 'Figures', 'liquid_biplot.png'))


import PCA_analysis as PC_custom

#make the PC plots
PC_custom.PC12_plots(PC_df, [], rep, directoryL, 'tif')
test = PC_custom.PC_av(PC_df, [])
PC_custom.PC_traj(test, rep, directoryL, 'tif')


#%% now onto the stats

#for this it is usful to append the conditions onto the dataframe
for rep in featuresL2:
    featuresL2 [rep] ['drug'] = drugL[rep]
    featuresL2[rep] ['concentration'] = concL[rep]
    #featuresA2[rep]['exp'] =exp_namesA[rep]
    featuresL2[rep] ['date'] = dateL[rep]
    
#compare each compound to control data
controlMeans = {}
for rep in featuresL2:
    controlMeans[rep] = pd.DataFrame()
    for line in range(len(featuresL2[rep])):
        if featuresL2[rep]['drug'].iloc[line] == 'DMSO':
            DMSO = featuresL2[rep].iloc[line]
            controlMeans[rep] = controlMeans[rep].append(DMSO.to_frame().transpose())
            del DMSO
    controlMeans[rep] = controlMeans[rep].reset_index(drop=True)

#so can now compile a matrix of p_values comparing each drug to control means
    #need to loop through the different drugs and basically just compare all the features to controlMeans
pVals = {}
feats =list(featuresL2[rep].columns)
feats = feats
for rep in featuresL2:
    pVals[rep] = pd.DataFrame()
    for drug in uniqueDrugs:
        if drug == 'DMSO':
            continue
        else:        
            currentInds = pd.DataFrame()
            for line in featuresL2[rep].iterrows():
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
                    for feature in currentInds.columns[0:-3]:
                        test.append(stats.ttest_ind(testing[feature], controlMeans[rep][feature]))
       
                    ps = [(test[i][1]) for i in range(len(test))] #make into a list
                    ps.append(drug)
                    ps.append(dose)
                    ps.append(dateL[rep]['date'][1])
        
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
        reg, corrP, t,s =  smm.multipletests(pVals[rep].values[cond, :-3],\
                                             alpha=0.05, method = 'fdr_bh', \
                                             is_sorted = False, returnsorted = False)
        corrP = list(corrP)
        corrP.append(pVals[rep]['drug'].iloc[cond])
        corrP.append(pVals[rep]['concentration'].iloc[cond])
        corrP = pd.DataFrame(corrP).transpose()
        bh_p [rep]= bh_p[rep].append(corrP)
        del corrP
        #add in the feature names
    bh_p[rep].columns = feats[:-1]
    bh_p [rep]= bh_p[rep].reset_index(drop = True)

    #now can filter out the features that have no significant p-values
    top_feats[rep]= bh_p[rep].values[:,0:-3] <=0.05 #0.05 significance level
    post_exclude [rep]= [] #initialise
    sig_feats[rep] = []
    for feat in range(bh_p[rep].shape[1]-3):
        if np.sum(top_feats[rep][:,feat]) ==0:
            post_exclude[rep].append(bh_p[rep].columns[feat]) #all the features that show no difference
        else:
            sig_feats[rep].append((bh_p[rep].columns [feat], np.sum(top_feats[rep][:,feat])))
    
    #sort by most to least
    sig_feats[rep].sort(key =lambda tup:tup[1])
    sig_feats[rep].reverse()

#make some violin plots of the significant features
import feature_swarms as swarm

cmap =  sns.color_palette("tab20", len(uniqueDrugs))
sns.set_style('whitegrid')

for rep in featuresL2:
    for feat in range(0,10):
        swarm.swarms (rep, sig_feats[rep][feat][0], featuresL2[rep], directoryL, '.tif', cmap)


#so looks like can pull out differences between DMSO and chloropromazine and clozapine. Data is messier probably
        #because only one repeat

#%% classifier v1
    #use only features that are significant to train the classifier
        
#convert the list of tuples into a list of features
bh_list = [i[0] for i in sig_feats[rep]]
#only 120 features

featMatAllL3 =  featMatAllL[bh_list]
featMatAllL3['drug'] = drug_allL
featMatAllL3['concentration'] = conc_allL
featMatAllL3['date'] = date_allL


#just try linear discriminant analysis on this reduced feature set
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#make a conditions dataframe
conds = pd.concat([featMatAllL3['drug'], featMatAllL3['concentration']], axis =1)

#make this condition so just clozapine at 10um vs all the others - 2 conditions
    #still include original drug
conds2 = pd.DataFrame()
for line in conds.iterrows():
    line2 = line[1].to_frame().transpose()
    if line[1]['drug'] == 'Clozapine' and line[1]['concentration'] == 10:
        line2['drug'] = 'Clozapine10'
        line2['class'] = 'Clozapine10'
    else:
        line2 ['drug'] =line[1]['drug']
        line2['class'] = 'Other'
    conds2 = conds2.append(line2)
    del line2

#make into arrays for classification
X = np.array(featMatAllL3.iloc[:,:-3])
y = np.array(conds2['class'])

#for splitting the data and crossvalidation
    #shuffle 10-fold cross validation
sss = StratifiedShuffleSplit(n_splits = 10, train_size=0.5)

#set liniar discriminant classifier
clf = LinearDiscriminantAnalysis(n_components = None, priors = None, \
                                 shrinkage = 'auto', solver = 'eigen')

#initialise lists for storing the multiple rounds of cross validation
score = []
class_scaling = []
class_means = []
X3=[]
coefs=[]
#start cross-validation
for train_index, test_index in sss.split(X, y):
    #split the testing and training data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
   
    #then do LDA using these training/testing sets
        #store the transformed scaling for each set
    X3.append(clf.fit_transform(X_train, y_train))
    
    #store the classification score
    score.append(clf.score(X_test, y_test))
   
    #weightings for each feature for each run
       #clf.classes_ = 'Clozapine10', 'other'
    coefs.append(clf.coef_[0])
    
    #scaling matrix - from PCA of feature space
    class_scaling.append(clf.scalings_)
    
    #mean (raw) for each class
    class_means.append(clf.means_)

    del X_train, X_test, y_train, y_test

classes = clf.classes_
del clf

#v high classification score
    #rank the features and produce a heatmap of the
""" now want to pick out the top features that contribute to each round of CV-LDA need 
to pick out the top and bottom ranking features"""

#rank the coef scores for each round of validation
temp = []
for i in coefs:
    temp.append(np.array(stats.rankdata(i)))

#make into a dataframe
feat_scores = pd.DataFrame(data = np.array(temp), columns = bh_list)
del temp

#now find out which features are in the top and bottom 10%, and then only take forward \
#those that are in the top/bottom 10% in 50% of the CVs
bottom = list(feat_scores.columns[np.sum(feat_scores<=int(feat_scores.shape[1]/10))>=5]) #bottom 10%
top = list(feat_scores.columns[np.sum(feat_scores>=int(feat_scores.shape[1] - feat_scores.shape[1]/10))>=5]) #top 10%

#combine
combi = top + bottom 
#11 features

#make a dataframe of this final featurespace
final_feat = pd.concat([featMatAllL3[combi], featMatAllL3.iloc[:,-3:]], axis=1)
final_feat['class'] = conds2['drug']

#make a colormap to assign colours - based on class (ie clozapine10 is separate)
cmap1 = sns.color_palette("tab20", np.unique(final_feat['class']).shape[0])

#plot these as swarms:
import feature_swarms as swarm
for item in combi:
    swarm.swarms('all', item, featMatAllL3, directoryL, '.tif', cmap1)
    plt.close()




#make a clustergram
    #1. make lut for drug colors
    #2. map the lut onto the clustergram
lut = dict(zip(np.unique(final_feat['class']), cmap1))

#add in row colors to the dataframe
row_colors = final_feat['class'].map(lut)#map onto the feature Matrix

#make clustergram
cg=sns.clustermap(final_feat.iloc[:,:-4], metric  = 'euclidean', cmap = 'inferno', \
                  row_colors = row_colors)
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
         (final_feat['class'][cg.dendrogram_row.reordered_ind]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8)
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) 
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
#save fig
plt.savefig(os.path.join(directoryL[0:-7], 'Figures', 'Liquid_LDA_clustergram1.tif'), \
            dpi =150, bbox_inches = 'tight', pad_inches = 1)
plt.show()

#make list of the final order of the drugs
drug_order = list(final_feat['class'][cg.dendrogram_row.reordered_ind])
conc_order = list(final_feat['concentration'][cg.dendrogram_row.reordered_ind])
feat_order =list(final_feat.iloc[:,:-4].columns[cg.dendrogram_col.reordered_ind])

#make a figure of the colorbar
colors = [(v) for k,v in lut.items()]
    #plot separately
plt.figure()
ax = plt.imshow([colors])
ax.axes.set_xticklabels(range(0,11,1))
ax.axes.set_xticklabels(lut.keys(), rotation = 90)
ax.axes.set_xticks(np.arange(0,10,1))
ax.axes.xaxis.set_ticks_position('top')
plt.savefig(os.path.join(directoryL[0:-7], 'Figures', 'drug_colors.png'),\
            bbox_inches='tight',dpi =150)
plt.close()

del combi, top, bottom

#%% now do mRMR on entire liquids data set and see how that fares in the classifier

import pymrmr

#need to discretise the data prior to implementing the algorithm
    #kernel density function can do this
        #alternative is to use pandas cut function - so bin data using
        #The Freedman-Diaconis Rule says that the optimal bin size of a histogram is
        # Bin Size=2⋅IQR(x)n^(−1/3)
        #and then divide this by the total range of the data to determine the number of bins

#use histogram blocks - freedman diaconis
bin_cutoff = {}
for feat in featMatAllL2.columns:
    plt.ioff()
    bin_cutoff[feat] = np.histogram(featMatAllL2[feat], bins='fd')[1]

#use these to create bins for the cutting up the data - can input into pandas cut
cat = pd.DataFrame()
for feat in featMatAllL2.columns:
    cat[feat]=pd.cut(featMatAllL2[feat], bins= bin_cutoff[feat], \
       labels = np.arange(1,len(bin_cutoff[feat])), include_lowest=True)

#make ints
cat2 = pd.DataFrame(data = np.array(cat.values), dtype = int, columns = cat.columns)        
#add in info about rows
cat.insert(0, column = 'drug', value = featMatAllL['drug'], allow_duplicates=True)

#select 150 features using mRMR
mrFeatsL = pymrmr.mRMR(cat2, 'MID', 150)

#export these features as txt tile
out = open(os.path.join(directoryA[:-7], 'mRMR_feats.txt'), 'w')
out.writelines(["%s\n" % item  for item in mr_Feats])
out.close()


