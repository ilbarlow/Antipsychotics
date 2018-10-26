#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:46:08 2018

@author: ibarlow
"""

""" PCA implementation; comparison of mRMR feature selection vs significant feature selection for 
running LDA"""

import TierPsyInput as TP
import numpy as np
import pandas as pd

directoryA, fileDirA, featuresA,trajectoriesA =  TP.TierPsyInput('new', 'Liquid')

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
fid_eggs = os.path.join (directoryA[:-7], 'ExtraFiles', 'egg_all.csv')
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
    
#%% combine the data

import numpy as np

#data normalization
#combine the Z-scored data as in the tSNE - this is better for when combining all the experiments
     #first impute the nans in the Z-scored data

featMatTotalNorm_mean = {}
for rep in featuresZ1:
    featMatTotalNorm_mean[rep] = featuresZ1[rep].fillna(featuresZ1[rep].mean(axis = 0))

featuresZ2 = {}
featMatAll = pd.DataFrame()
for rep in featMatTotalNorm_mean:
    featuresZ2 [rep] = pd.concat([featMatTotalNorm_mean[rep], drugA2[rep], concA2[rep], dateA2[rep]], axis =1)
    featMatAll = featMatAll.append(featuresZ2[rep])

#reset index
featMatAll = featMatAll.reset_index(drop = True)
featMatAll2 = featMatAll.copy()

drug_all = featMatAll2.pop ('drug')
conc_all = featMatAll2.pop('concentration')
date_all = featMatAll2.pop ('date')

reps = list(featuresZ2.keys())


#save this feature matrix
featMatAll.to_csv(os.path.join(os.path.dirname(directoryA), 'NewFeatures.csv'))
#%% so now onto the PCA
    #use sklearn toolkit as more reliable, faster and more concise than my own code

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#load data again if necessary
FeatIn = '/Volumes/behavgenom_archive$/Adam/screening/antipsychotics/Figures/New_Features/NewFeatures.csv'

featMatAll2 = pd.read_csv(FeatIn, index_col=0)
drug_all = featMatAll2.pop ('drug')
conc_all = featMatAll2.pop('concentration')
date_all = featMatAll2.pop ('date')

allDrugs = np.unique(drug_all)

#make array of z-scored data
X = np.array(featMatAll2.select_dtypes(include='float'))
#initialise PCA
pca = PCA()
X2= pca.fit_transform(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
thresh = cumvar <= 0.95 #set 95% variance threshold
cut_off = int(np.argwhere(thresh)[-1])

#make a plot
sns.set_style('whitegrid')
plt.plot(range(0, len(cumvar)), cumvar*100)
plt.plot([cut_off, cut_off], [0, 100], 'k')
plt.xlabel('Number of Principal Components', fontsize =16)
plt.ylabel('variance explained', fontsize =16)
plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'agarPCvar.png'), dpi =150)
plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'agarPCvar.svg'),dpi = 150)

#now put the 1:cut_off PCs into a dataframe
PCname = ['PC_%d' %(p+1) for p in range (0,cut_off+1)]
PC_df = pd.DataFrame(data= X2[:,:cut_off+1], columns = PCname)
PC_df['drug'] = drug_all
PC_df['concentration'] = conc_all
PC_df['date'] = date_all

#components that explain the variance
    #make a dataframe ranking the features for each PC and also include the explained variance (z-normalised)
        #in separate dataframe called PC_sum
PC_feat = [] #features
PC_sum =[] #explained variance
for PC in range(0, len(PCname)):
    PC_feat.append(list(featMatAll2.columns[np.argsort(pca.components_[PC])]))
    PC_sum.append(list((pca.components_[PC])/ np.sum(abs(pca.components_[PC]))))
     #                   np.mean((pca.components_[PC])))/np.std((pca.components_[PC])))
    
#dataframe containing standards scores of contibution of each feature
        #rows correspond to PCs
PC_vals = pd.DataFrame(data= PC_sum, columns =  featMatAll2.columns)

#okay so now can plot as biplot
plt.figure()
for i in range(0,1):
    plt.arrow(0,0, PC_vals.iloc[0,:][PC_feat[0][-1-i]]*100, \
              PC_vals.iloc[1,:][PC_feat[0][-1-i]]*100,color= 'b')
    plt.arrow(0,0, PC_vals.iloc[0,:][PC_feat[1][-1-i]]*100,\
              PC_vals.iloc[1,:][PC_feat[1][-1-i]]*100, color='r')
    plt.text(PC_vals.iloc[0,:][PC_feat[0][-1-i]] + 0.7,\
             PC_vals.iloc[1,:][PC_feat[0][-1-i]] - 0.3, PC_feat[0][-1-i],\
             ha='center', va='center')
    plt.text(PC_vals.iloc[0,:][PC_feat[1][-1-i]]+0.5, PC_vals.iloc[1,:][PC_feat[1][-1-i]]+1,\
         PC_feat[1][-1-i], ha='center', va='center')

plt.xlim (-2, 2)
plt.ylim (-2, 2)
plt.xlabel('%' + 'PC_1 (%.2f)' % (pca.explained_variance_ratio_[0]*100), fontsize = 16)
plt.ylabel('%' + 'PC_2 (%.2f)' % (pca.explained_variance_ratio_[1]*100), fontsize = 16)
plt.show()
plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'agar_biplot.png'))


import PCA_analysis as PC_custom 
cmap1 = sns.color_palette("tab20", len(np.unique(drug_all))+1) #need this to match the clustergram from mRMR so add one for cloz10
#get rid of 5th row, which woudl be cloz10 -  there is probably a smarter way to do this...
cmap1 = np.delete(cmap1, 4, axis = 0)

#make the PC plots
PC_custom.PC12_plots(PC_df, [], 'all' ,cmap1, directoryA, 'tif', 'concentration')
PCmean, PCsem = PC_custom.PC_av(PC_df, [], 'concentration')
PC_custom.PC_traj(PCmean, PCsem,['PC_1', 'PC_2'], 'all', directoryA, 'tif', cmap1, [], start_end=False)

#the sklearn and my custom PCA gave exactly the same results - Phew

#updated PC12 plots
import PC_traj as PCJ
from matplotlib.colors import LinearSegmentedColormap
import make_colormaps as mkc

cmapGraded = [] #and graded colormaps
for item in cmap1:
    cmapGraded.append([(1,1,1), (item)])

lutGraded = dict(zip(allDrugs, cmapGraded))
cm={}
for drug in lutGraded:
    cmap_name = drug
    # Create the colormap
    cm[drug] = LinearSegmentedColormap.from_list(
        cmap_name, lutGraded[drug], N=60)
    plt.register_cmap(cmap = cm[drug])    
    
#plot the colorgradients
mkc.plot_color_gradients(cm, cm.keys())
plt.savefig(os.path.join(os.path.dirname(directoryA), 'Figures', 'gradientDrugColors.png'))

PCJ.PC_trajGraded(PCmean, PCsem, 'all', directoryA, 'tif', 'concentration', start_end=False, cum_var = cumvar, legend= 'off' )

#%% now on to the stats
    #for this it is usful to append the conditions onto the dataframe
for rep in featuresEA_1:
    featuresEA_1 [rep] ['drug'] = drugA2[rep]
    featuresEA_1[rep] ['concentration'] = concA2[rep]
    #featuresA2[rep]['exp'] =exp_namesA[rep]
    featuresEA_1[rep] ['date'] = dateA2[rep]
    
#compare each compound to control data
controlMeans = {}
for rep in featuresEA_1:
    controlMeans[rep] = pd.DataFrame()
    for line in range(len(featuresEA_1[rep])):
        if featuresEA_1[rep]['drug'].iloc[line] == 'DMSO':
            DMSO = featuresEA_1[rep].iloc[line]
            controlMeans[rep] = controlMeans[rep].append(DMSO.to_frame().transpose())
            del DMSO
    controlMeans[rep] = controlMeans[rep].reset_index(drop=True)

#so can now compile a matrix of p_values comparing each drug to control means
    #need to loop through the different drugs and basically just compare all the features to controlMeans
pVals = {}
feats =list(featuresEA_1[rep].columns)
feats = feats
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
                    ps.append(dateA2[rep][1])
        
                    temp = pd.DataFrame(ps).transpose()
                    pVals[rep] = pVals[rep].append(temp)
                    del temp, to_test, testing
            del currentInds

    #add in features
    pVals[rep].columns = feats
    pVals[rep] = pVals[rep].reset_index (drop=True)   

#import module for multiple comparison correction
import statsmodels.stats.multitest as smm

#now correct for multiple comparisons - bejamini hochberg procedure
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
    top_feats[rep]= bh_p[rep].values[:,0:-2] <=0.05 #0.05 significance level
    top_feats[rep] = pd.DataFrame(data=top_feats[rep], columns = bh_p[rep].iloc[:,:-2].columns)
    top_feats[rep] = pd.concat([top_feats[rep], bh_p[rep].iloc[:,-2:]], axis=1 )
    
    post_exclude [rep]= [] #initialise
    sig_feats[rep] = []
    for feat in range(bh_p[rep].shape[1]-2):
        if np.sum(top_feats[rep].iloc[:,feat]) ==0:
            post_exclude[rep].append(bh_p[rep].columns[feat]) #all the features that show no difference
        else:
            sig_feats[rep].append((bh_p[rep].columns [feat], np.sum(top_feats[rep].iloc[:,feat])))
    
    #sort by most to least
    sig_feats[rep].sort(key =lambda tup:tup[1])
    sig_feats[rep].reverse()

#make a list of which features are significantly different for each drug
pVals2 = {}
for rep in sig_feats:
    pVals2[rep] = pd.DataFrame()
    for drug in uniqueDrugs:
        temp = top_feats[rep][top_feats[rep]['drug']==drug]
        temp2 = bh_p[rep][bh_p[rep]['drug'] == drug]
        conc = np.unique(temp['concentration'])
        for c in conc:
            temp3 = temp[temp['concentration']==c]
            temp4 = temp2[temp2['concentration'] ==c]
            #feats = list(temp3.columns[np.where(temp3)[1]])
            ps = temp4[temp3.columns[np.where(temp3)[1]]]
            ps['concentration']=  c
            ps['drug'] = drug
            pVals2[rep] = pVals2[rep].append(ps)
            del ps, temp3, temp4
    del temp, temp2, conc

#export the pVals2 as a csv
for rep in pVals2:
    pVals2[rep].to_csv(os.path.join(directoryA[:-7], rep + 'sigPs.csv' ))

#make some violin plots of the significant features
import feature_swarms as swarm

cmap =  sns.color_palette("tab20", len(uniqueDrugs))
sns.set_style('whitegrid')

for rep in featuresA2:
    for feat in range(0,10):
        swarm.swarms (rep, sig_feats[rep][feat][0], featuresA2[rep], directoryA, '.tif', 'Blues')

#make a list of any of the features that are significantly different in all experiments
stats_feats = []
for rep in sig_feats:
    stats_feats += list(list(zip(*sig_feats[rep]))[0])
stats_feats = np.unique(stats_feats)

#%% reduce the feature set to the significant ones - for Clozapine10

#make a list combined from all reps
cloz10Feats = []
for rep in pVals2:
    cloz10 = pVals2[rep][pVals2[rep]['drug'] == 'Clozapine'][pVals2[rep]['concentration']==10.0]
    cloz10Feats += list(cloz10.columns[np.where(cloz10.notnull())[1]])
    del cloz10
    
cloz10Feats = np.unique(cloz10Feats)
#264 features that are significantly different for cloz10 across all experiments

sFeatMatAll = featMatAll[cloz10Feats]


#use these features for classification
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

#make a conditions dataframe
conds = pd.concat([sFeatMatAll.pop('drug'), sFeatMatAll.pop('concentration')], axis =1)

#make this condition so just clozapine at 10um vs all the others - 2 conditions
    #still include original drug
    #include extra columns with number as classification
conds2 = pd.DataFrame()
for line in conds.iterrows():
    line2 = line[1].to_frame().transpose()
    if line[1]['drug'] == 'Clozapine' and line[1]['concentration'] == 10:
        line2['drug'] = 'Clozapine10'
        line2['class'] = 'Clozapine10'
        line2['class2'] = 1
    else:
        line2 ['drug'] =line[1]['drug']
        line2['class'] = 'Other'
        line2['class2'] = 0
    conds2 = conds2.append(line2)
    del line2

#use label encoder to make drugs numerical
from sklearn.preprocessing import LabelEncoder

#train encoder
drugLabels = conds['drug'].values
enc = LabelEncoder()
label_encoder = enc.fit(drugLabels)
drugNums = label_encoder.transform(drugLabels) + 1

#append into the dataframe
conds2['drug2'] = drugNums

#and add in classification as integeter according to drug
conds3 = pd.DataFrame()
for line in conds.iterrows():
    line2 = line[1].to_frame().transpose()
    #now loop through to assign drugs
    for drug in range(0, len(uniqueDrugs)):
        if line[1]['drug'] == uniqueDrugs[drug]:
            line2['drugInt'] = drug
            conds3 = conds3.append(line2)
        else:
            continue
    del line2

#make into arrays for classification
X = np.array(sFeatMatAll)
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

""" now want to pick out the top features that contribute to each round of CV-LDA need 
to pick out the top and bottom ranking features"""

#rank the coef scores for each round of validation
temp = []
for i in coefs:
    temp.append(np.array(stats.rankdata(i)))

#make into a dataframe
feat_scores = pd.DataFrame(data = np.array(temp), columns = sFeatMatAll.columns)
del temp

#now find out which features are in the top and bottom 10%, and then only take forward \
#those that are in the top/bottom 10% in 50% of the CVs
bottom = list(feat_scores.columns[np.sum(feat_scores<=int(feat_scores.shape[1]/10))>=5]) #bottom 10%
top = list(feat_scores.columns[np.sum(feat_scores>=int(feat_scores.shape[1] - feat_scores.shape[1]/10))>=5]) #top 10%

#combine
combi = top + bottom 
#22 features
print(str(len(combi)) + ' unique features')

sfinal_feat= pd.concat([sFeatMatAll[combi], conds2], axis =1)

#make a colormap to assign colours - based on class (ie clozapine10 is separate)
cmap1 = sns.color_palette("tab20", np.unique(sfinal_feat['drug']).shape[0])

#make a clustergram
    #1. make lut for drug colors
    #2. map the lut onto the clustergram
lut = dict(zip(np.unique(sfinal_feat['drug']), cmap1))

#add in row colors to the dataframe
row_colors = sfinal_feat['drug'].map(lut)#map onto the feature Matrix

#make clustergram
cg=sns.clustermap(sfinal_feat.iloc[:,:-3], metric  = 'euclidean', cmap = 'inferno', \
                  row_colors = row_colors)
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
         (sfinal_feat['drug'][cg.dendrogram_row.reordered_ind]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8)
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) 
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
#save fig
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'Agar_stats_LDA_clustergram1.tif'), \
            dpi =150, bbox_inches = 'tight', pad_inches = 1)
plt.show()

#make list of the final order of the drugs
drug_order = list(sfinal_feat['drug'][cg.dendrogram_row.reordered_ind])
conc_order = list(sfinal_feat['concentration'][cg.dendrogram_row.reordered_ind])
feat_order =list(sfinal_feat.iloc[:,:-3].columns[cg.dendrogram_col.reordered_ind])

import csv

with open(os.path.join(directoryA[:-7], 'drugOrder.csv'), 'w') as fid:
    wr = csv.writer(fid, dialect='excel')
    wr.writerow(drug_order)

with open(os.path.join(directoryA[:-7], 'feat_order.csv'), 'w') as fid:
    wr = csv.writer(fid, dialect = 'excel')
    wr.writerow(feat_order)
    
    
#make a figure of the colorbar
colors = [(v) for k,v in lut.items()]
    #plot separately
plt.figure()
ax = plt.imshow([colors])
ax.axes.set_xticklabels(range(0,11,1))
ax.axes.set_xticklabels(lut.keys(), rotation = 90)
ax.axes.set_xticks(np.arange(0,10,1))
ax.axes.xaxis.set_ticks_position('top')
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'drug_colors.png'),\
            bbox_inches='tight',dpi =150)
plt.close()

del combi, top, bottom, drug_order, conc_order, feat_order, X2, X3, X, bh_p


#%% okay so now do an implementation using mRMR and LDA
    #1. Histogram bins determined using 'freedman-diaconis' rule
    #2. Use this to select 150 features for LDA
    #3. Compare the LDA scores to those from 150 randomly selected features
    #4. Then select top 20% accoriding to LDA score and plot the hierachical clustergram

import pymrmr

#need to discretise the data prior to implementing the algorithm
        # use pandas cut function - so bin data using
        #The Freedman-Diaconis Rule says that the optimal bin size of a histogram is
        # Bin Size=2⋅IQR(x)n^(−1/3) and then divide this by the total range of 
        # the data to determine the number of bins

#use histogram blocks - freedman diaconis
bin_cutoff = {}
for feat in featMatAll2.columns:
    plt.ioff()
    bin_cutoff[feat] = np.histogram(featMatAll2[feat], bins='fd')[1]

#use these to create bins for the cutting up the data - can input into pandas cut
cat = pd.DataFrame()
for feat in featMatAll2.columns:
    cat[feat]=pd.cut(featMatAll2[feat], bins= bin_cutoff[feat], \
       labels = np.arange(1,len(bin_cutoff[feat])), include_lowest=True)

#make ints and copy
cat2 = pd.DataFrame(data = np.array(cat.values), dtype = int, columns = cat.columns)        

#add in info about rows
cat.insert(0, column = 'drug', value = conds2['drug2'], allow_duplicates=True)
cat2.insert(0, column = 'class2', value = conds2['class2'], allow_duplicates = True)

#select 150 features using mRMR
    #based on all drugs
mrFeatsA = pymrmr.mRMR(cat, 'MID', 150)

#based on just clozapine 10 vs all
mrFeatsA2 = pymrmr.mRMR(cat2, 'MID', 150)

#make a temporary directory for figures
directoryTemp = '/Volumes/behavgenom_archive$/Adam/screening/antipsychotics/Figures'

#export these features as txt tile
out = open(os.path.join(directoryTemp[:-7], 'mRMR_featsAgarAllclassesFINAL.txt'), 'w')
out.writelines(["%s\n" % item  for item in mrFeatsA])
out.close()

#and for clozapine only
out = open(os.path.join(directoryTemp[:-7], 'mRMR_featsAgarClozFINAL.txt'), 'w')
out.writelines(["%s\n" % item  for item in mrFeatsA2])
out.close()


#plot swarms of top 10
#make some violin plots of the significant features
import feature_swarms as swarm

cmap =  sns.color_palette("tab20", len(uniqueDrugs))
sns.set_style('whitegrid')

#make some swarm plots
for feat in range(0,10):
    swarm.swarms ('all', mrFeatsA[feat], featMatAll, directoryTemp, '.tif', cmap)

#and for clozapine only
#make some swarm plots
for feat in range(0,10):
    swarm.swarms ('all', mrFeatsA2[feat], featMatAll, directoryTemp, '.tif', cmap)
plt.show()  

#so this is the mRMR selected feature set.
mrFeatMatAll = pd.concat([featMatAll[mrFeatsA], featMatAll.iloc[:,-3:]], axis=1)


#%%
# We want to show that this feature set performs better than a random set of features
    #so use sss split again to pick out 150 features randomly, and do LDA (10CV'd)
        #need several loops to do this one

#first perform LDA using 10-fold cross validation
     #use the conditions dataframe already made previously

#make into arrays for classification
#X = np.array(mrFeatMatAll.iloc[:,:100]) - this is old

#array of drug classes
y = np.array(conds2['class'])

#for splitting the data and crossvalidation
    #shuffle 10-fold cross validation
    #split data 50% train, 50% test
sss = StratifiedShuffleSplit(n_splits = 10, train_size=0.5, test_size = 0.5)

#set liniar discriminant classifier
clf = LinearDiscriminantAnalysis(n_components = None, priors = None, \
                                 shrinkage = 'auto', solver = 'eigen')

#how to decide how many features to use?
    #initiate lists to compare 10 -150 features from mRMR
splitsM = {}
scoreM = {}
coefsM = {}
classScalingM = {}
#cvScore = {}
for featNo in range (1,80,1):
    splitsM[featNo] = mrFeatMatAll.iloc[:,:featNo]
    Xm = np.array(splitsM[featNo].values)
    
    scoreM[featNo] =[]
    coefsM[featNo] = []
    classScalingM[featNo] = []
    #cvScore = []
    #now train the classifier
    for train, test in sss.split(Xm,y):
        #split the testing and training data
        X_train, X_test = Xm[train], Xm[test]
        y_train, y_test = y[train], y[test]
   
        #then do LDA using these training/testing sets
        #store the transformed scaling for each set
        clf.fit(X_train, y_train)
    
        #store the classification score
        scoreM[featNo].append(clf.score(X_test, y_test))
   
        #weightings for each feature for each run - for clozapine10
        coefsM[featNo].append(clf.coef_[0])
    
            #scaling matrix - from PCA of feature space
        classScalingM[featNo].append(clf.scalings_)
        
        #cvScore.append(cross_val_score(clf, X_train, y_test))
        
        del X_train, X_test, y_train, y_test, train, test 
    scoreM[featNo] = np.array(scoreM[featNo])
    del Xm
 
#compile for plotting
scoreMav = []
scoreMstd = []
scoreMsem = []
for featNo in scoreM:
    scoreMav.append(scoreM[featNo].mean())
    scoreMstd.append(scoreM[featNo].std())
    scoreMsem.append(scoreM[featNo].std()/np.sqrt(len(scoreM[featNo])))
    
sns.set_style('whitegrid')
plt.errorbar(scoreM.keys(), scoreMav, yerr= scoreMstd)
plt.title('Classification score with mRMR selected features')
plt.ylabel('Score')
plt.xlabel ('number of features selected')
plt.ylim([0.9, 1])
plt.show()

#compare to randomly selected features
#set parameters again
import random
#use featMatAll2 columns as features
Fy = list(featMatAll2.columns)

sss = StratifiedShuffleSplit(n_splits = 10, train_size=0.5, test_size = 0.5)

#and for LDA
clf = LinearDiscriminantAnalysis(n_components = None, priors = None, \
                                 shrinkage = 'auto', solver = 'eigen')

#initiate lists to compare - R stands for Random
splitsR = {}
scoreR = {}
coefsR = {}
classScalingR = {}
scoreRav = {}
for featNo in range (1,80,1):
    splitsR[featNo] = []
    scoreR[featNo] ={}
    coefsR[featNo] = {}
    classScalingR[featNo] = {}
    scoreRav[featNo] = []
    temp = []
    for i in range(0,10):
        splitsR[featNo].append(random.sample(Fy, featNo))
    
        Xr = np.array(featMatAll[splitsR[featNo][i]])
    
        scoreR[featNo][i] = []
        coefsR[featNo][i] = []
        classScalingR[featNo][i] = []

        #now train the classifier
        for train, test in sss.split(Xr,y):
            #split the testing and training data
            X_train, X_test = Xr[train], Xr[test]
            y_train, y_test = y[train], y[test]
   
            #then do LDA using these training/testing sets
            #store the transformed scaling for each set
            
            clf.fit(X_train, y_train)
    
            #store the classification score
            scoreR[featNo][i].append(clf.score(X_test, y_test))
   
            #weightings for each feature for each run - for clozapine10
            coefsR[featNo][i].append(clf.coef_[0])
    
            #scaling matrix - from PCA of feature space
            classScalingR[featNo][i].append(clf.scalings_)

            del X_train, X_test, y_train, y_test, train, test 
        scoreR[featNo][i] = np.array(scoreR[featNo][i])
        temp.append(np.mean(scoreR[featNo][i]))
        
        del Xr
    
    scoreRav[featNo].append(temp)
    del temp

#make an array of score averages to plot
scoreRav2 = []
scoreRstd2=[]
scoreRsem = []
for featNo in scoreRav.keys():
    scoreRav2.append(np.mean(scoreRav[featNo]))
    scoreRstd2.append(np.std(scoreRav[featNo]))
    scoreRsem.append(np.std(scoreRav[featNo]/np.sqrt(len(scoreRav[featNo][0]))))

#now make a plot comparing the mRMR to randomly selected features
sns.set_style('whitegrid')
plt.errorbar(scoreRav.keys(), (1-np.array(scoreRav2))*100, yerr= np.array(scoreRsem)*100)
plt.errorbar(scoreM.keys(), (1-np.array(scoreMav))*100, yerr= np.array(scoreMsem)*100, color = 'r')
plt.legend(['Random features', 'mRMR selected features'], loc= 'best', fancybox = True)
plt.title('Classification score mRMR vs Random features')
plt.ylabel('Error (%)')
plt.xlabel ('number of features selected')
plt.ylim([0, 8])
plt.xlim([0, 80])
plt.savefig(os.path.join(directoryTemp[:-7], 'Figures', 'mMRMRvRandom_LDAscores.tif'), dpi = 200)
plt.show()

#now can just take the top 30 features
featNo = 30

#cmap already defined
    #map onto the drug classes
mrFeatMatFinal = pd.concat([splitsM[featNo], conds2], axis=1)

#add in row colors to the dataframe
row_colors = mrFeatMatFinal['drug'].map(lut)#map onto the feature Matrix

#make clustergram
cg=sns.clustermap(mrFeatMatFinal.iloc[:,:-4], metric  = 'euclidean', cmap = 'inferno', \
                  row_colors = row_colors)
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
         (mrFeatMatFinal['drug'][cg.dendrogram_row.reordered_ind]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8) #y tick labels
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) #x tick labels
#set position
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
#save fig
plt.savefig(os.path.join(directoryTemp[0:-7], 'Figures', 'AgarLDA_clustergramMRMR' + str(featNo) + '.tif'), \
            dpi =150, bbox_inches = 'tight', pad_inches = 1)
plt.show()

#make list of the final order of the drugs
drug_order = list(mrFeatMatFinal['drug'][cg.dendrogram_row.reordered_ind])
conc_order = list(mrFeatMatFinal['concentration'][cg.dendrogram_row.reordered_ind])
feat_order =list(mrFeatMatFinal.iloc[:,:-4].columns[cg.dendrogram_col.reordered_ind])

with open(os.path.join(directoryTemp[:-7], 'drugOrderMRMR'+ str(featNo) + '.csv'), 'w') as fid:
    wr = csv.writer(fid, dialect='excel')
    wr.writerow(drug_order)

with open(os.path.join(directoryTemp[:-7], 'featOrderMRMR' + str(featNo) + '.csv'), 'w') as fid:
    wr = csv.writer(fid, dialect = 'excel')
    wr.writerow(feat_order)

#make a figure of the colorbar
colors = [(v) for k,v in lut.items()]

#plot separately
sns.set_style('white')
plt.figure()
ax = plt.imshow([colors])
ax.axes.set_xticklabels(range(0,11,1))
ax.axes.set_xticklabels(lut.keys(), rotation = 90, fontsize = 16)
ax.axes.set_xticks(np.arange(0,10,1))
ax.axes.xaxis.set_ticks_position('top')
plt.savefig(os.path.join(directoryTemp[0:-7], 'Figures', 'drugColors.png'),\
            bbox_inches='tight',dpi =150)
plt.close()





#%% beyond here is old scrap





#rank the coef scores for each round of validation
temp = []
for i in coefsM[20]:
    temp.append(np.array(stats.rankdata(i)))

#make into a dataframe
feat_scores = pd.DataFrame(data = np.array(temp), columns = mrFeatsA[:20])
del temp

#now find out which features are in the top and bottom 10%, and then only take forward \
#those that are in the top/bottom 10% in 50% of the CVs
bottom = list(feat_scores.columns[np.sum(feat_scores<=int(feat_scores.shape[1]/10))>=5]) #bottom 10%
top = list(feat_scores.columns[np.sum(feat_scores>=int(feat_scores.shape[1] - feat_scores.shape[1]/10))>=5]) #top 10%

#combine
combi = top + bottom 
#18 features

#dataframe of final featurespace to make clustergram
mrFeatMatFinal = pd.concat([mrFeatMatAll[combi], conds2], axis=1)

#and make nice looking clustergram
    #lut already assigned

#add in row colors to the dataframe
row_colors = mrFeatMatFinal['drug'].map(lut)#map onto the feature Matrix

#make clustergram
cg=sns.clustermap(mrFeatMatFinal.iloc[:,:-3], metric  = 'euclidean', cmap = 'inferno', \
                  row_colors = row_colors)
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
         (mrFeatMatFinal['drug'][cg.dendrogram_row.reordered_ind]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8) #y tick labels
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) #x tick labels
#set position
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
#save fig
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'AgarLDA_clustergramMRMR.tif'), \
            dpi =150, bbox_inches = 'tight', pad_inches = 1)
plt.show()

#make list of the final order of the drugs
drug_order = list(mrFeatMatFinal['drug'][cg.dendrogram_row.reordered_ind])
conc_order = list(mrFeatMatFinal['concentration'][cg.dendrogram_row.reordered_ind])
feat_order =list(mrFeatMatFinal.iloc[:,:-3].columns[cg.dendrogram_col.reordered_ind])

with open(os.path.join(directoryA[:-7], 'drugOrderMRMR.csv'), 'w') as fid:
    wr = csv.writer(fid, dialect='excel')
    wr.writerow(drug_order)

with open(os.path.join(directoryA[:-7], 'featOrderMRMR.csv'), 'w') as fid:
    wr = csv.writer(fid, dialect = 'excel')
    wr.writerow(feat_order)

#make a figure of the colorbar
colors = [(v) for k,v in lut.items()]

#plot separately
sns.set_style('white')
plt.figure()
ax = plt.imshow([colors])
ax.axes.set_xticklabels(range(0,11,1))
ax.axes.set_xticklabels(lut.keys(), rotation = 90)
ax.axes.set_xticks(np.arange(0,10,1))
ax.axes.xaxis.set_ticks_position('top')
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'drugColors.png'),\
            bbox_inches='tight',dpi =150)
plt.close()

del combi, top, bottom

#%% now need to compare to 150 randomly selected features
import random

#use featMatAll2
Fy = list(featMatAll2.columns)
#same y used for classification as before

sss = StratifiedShuffleSplit(n_splits = 10, train_size=0.5, test_size = 0.5)
clf = LinearDiscriminantAnalysis(n_components = None, priors = None, \
                                 shrinkage = 'auto', solver = 'eigen')

#initiate lists to compare
splits = {}
scoreR = {}
coefsR = {}
classScaling = {}
scoreRav = {}
for featNo in range (10,150,10):
    splits[featNo] = []
    scoreR[featNo] ={}
    coefsR[featNo] = {}
    classScaling[featNo] = {}
    scoreRav[featNo] = []
    temp = []
    for i in range(0,11):
        splits[featNo].append(random.sample(Fy, featNo))
    
        Xt = np.array(featMatAll[splits[featNo][i]])
    
        scoreR[featNo][i] = []
        coefsR[featNo][i] = []
        classScaling[featNo][i] = []

        #now train the classifier
        for train, test in sss.split(Xt,y):
            #split the testing and training data
            X_train, X_test = Xt[train], Xt[test]
            y_train, y_test = y[train], y[test]
   
            #then do LDA using these training/testing sets
            #store the transformed scaling for each set
            
            clf.fit(X_train, y_train)
    
            #store the classification score
            scoreR[featNo][i].append(clf.score(X_test, y_test))
   
            #weightings for each feature for each run - for clozapine10
            coefsR[featNo][i].append(clf.coef_[0])
    
            #scaling matrix - from PCA of feature space
            classScaling[featNo][i].append(clf.scalings_)

            del X_train, X_test, y_train, y_test, train, test 
        scoreR[featNo][i] = np.array(scoreR[featNo][i])
        temp.append(np.mean(scoreR[featNo][i]))
        
        del Xt
    
    scoreRav[featNo].append(temp)
    del temp

#make an array of score averages to plot
scoreAv = []
scoreStd=[]
for featNo in scoreRav.keys():
    scoreAv.append(np.mean(scoreRav[featNo]))
    scoreStd.append(np.std(scoreRav[featNo]))
    
sns.set_style('whitegrid')
plt.errorbar(scoreRav.keys(), scoreAv, yerr= scoreStd)
plt.title('Classification score with randomly selected features')
plt.ylabel('Score')
plt.xlabel ('number of features selected')
plt.ylim([0.9, 1])
plt.show()
    

