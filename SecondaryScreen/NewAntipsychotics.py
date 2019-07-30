#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 12:03:27 2018

@author: ibarlow
"""

""" Analysis of New Antipsychotics Dataset collected on 20180906 -

New metadata storage format means that the filenames just have Set number and channel
so need to tie these details up with conditions """
%cd /Users/ibarlow/Documents/GitHub/pythonScripts/Functions/

import os
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
import displayfiles as df
import re
from scipy import stats

#generate list of features files
dirFeats, FeatFiles = df.displayfiles(validExt='featuresN.hdf5', inputfolder = None,\
                                      outputfile ='Featurefiles.tsv')

#load these features files
features = pd.DataFrame()
for item in FeatFiles:
    with pd.HDFStore(item, 'r') as fid:
        temp = fid['/features_stats'].pivot_table(columns = 'name', values = 'value')
        temp ['exp'] = os.path.basename(item)
        features = features.append(temp, sort=True)

features = features.reset_index(drop=True)

#remove features where all rows have the same value and interpolate nan features
to_remove = list(features.select_dtypes(include = 'float32').columns[features.std(axis=0)==0])
features = features.drop(columns = to_remove)
features = features.fillna(features.mean(axis=0))

#drop exp for zscoring 
featuresZ = features.select_dtypes(include = 'float32')
featuresZ = pd.DataFrame(stats.zscore(featuresZ, axis=0), columns = featuresZ.columns)

metadata = pd.read_csv(os.path.join(os.path.dirname(dirFeats),'AuxiliaryFiles', 'metadata.csv' ), index_col=False)

#now match up metadata with the files and add in drug, concentration information
    #can assume that both FeatFiles and metadata are sorted by set and channel
featuresZ2 = pd.DataFrame()
for i in range(0,featuresZ.shape[0]):
    update = featuresZ.iloc[i,:]
    update['drug'] = metadata[' drug type'][i]
    update['concentration'] = metadata[' drug concentration'][i]
    update ['date'] = str(metadata['date (YEARMODA)'][i])

    featuresZ2 = featuresZ2.append(update)
    del update

featuresZ2 = featuresZ2.reset_index(drop=True)

allDrugs = list(np.unique(featuresZ2['drug']))
allConcs = list(np.unique(featuresZ2['concentration']))

#make a list of the drug and concentration IDs
drugs = featuresZ2['drug']
concs = featuresZ2['concentration']

features = pd.concat([features, featuresZ2[['drug', 'concentration', 'date']]], axis=1)

#save the dataframe to a csv
featuresZ2.to_csv(os.path.join(os.path.dirname(dirFeats), 'FeaturesZ.csv'))
features.to_csv(os.path.join(os.path.dirname(dirFeats), 'FeatMatAllNewFeats.csv'),index=False)

#%% PCA on just this dataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#make array of z-scored data
X = np.array(featuresZ.values)
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
#plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'agarPCvar.png'), dpi =150)
#plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'agarPCvar.svg'),dpi = 150)

#now put the 1:cut_off PCs into a dataframe
PCname = ['PC_%d' %(p+1) for p in range (0,cut_off+1)]
PC_df = pd.DataFrame(data= X2[:,:cut_off+1], columns = PCname)
PC_df['drug'] = drugs
PC_df['concentration'] = concs
#PC_df['date'] = date_all

#components that explain the variance
    #make a dataframe ranking the features for each PC and also include the explained variance (z-normalised)
        #in separate dataframe called PC_sum
PC_feat = [] #features
PC_sum =[] #explained variance
for PC in range(0, len(PCname)):
    PC_feat.append(list(featuresZ.columns[np.argsort(pca.components_[PC])]))
    PC_sum.append(list((pca.components_[PC])/ np.sum(abs(pca.components_[PC]))))
     #                   np.mean((pca.components_[PC])))/np.std((pca.components_[PC])))
    
#dataframe containing standards scores of contibution of each feature
        #rows correspond to PCs
PC_vals = pd.DataFrame(data= PC_sum, columns =  featuresZ.columns)

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
#plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'agar_biplot.png'))

import PCA_analysis as PC_custom 
cmap1 = sns.color_palette("tab20", len(np.unique(featuresZ2['drug']))+1) #need this to match the clustergram from mRMR so add one for cloz10
#get rid of 5th row, which woudl be cloz10 -  there is probably a smarter way to do this...
cmap1 = np.delete(cmap1, 4, axis = 0)

#make the PC plots
PC_custom.PC12_plots(PC_df, [],[],cmap1,  dirFeats, 'tif', 'concentration')
PCmean, PCsem = PC_custom.PC_av(PC_df, [], 'concentration')
PC_custom.PC_traj(PCmean, PCsem,['PC_1', 'PC_2'], [], dirFeats, 'tif', cmap1,[], start_end = False)


