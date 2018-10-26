#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:56:14 2018

@author: ibarlow

Combining pesticides and antipsychotics
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

dirparent = '/Volumes/behavgenom$/Ida/Data/Antipsychotics/'
featfilename = 'FeatMatAllNewFeats.csv'

importfiles = []
for (root, dirs, files) in os.walk(dirparent):
        if files == []: #pass empty folders
            continue
        else:
            for file1 in files: #loop through to add the files
                if file1.endswith(featfilename):
                    importfiles.append(os.path.join(root, file1))
                else:
                    continue
featmats = {}

for item in importfiles:
    featmats [item.split('/')[-3]] = pd.read_csv(item, index_col= False)

featMatAll = pd.DataFrame()
for item in featmats:
    featMatAll = featMatAll.append(featmats[item], sort=True)
featMatAll = featMatAll.reset_index(drop = True)   

#remove features with too many nans
featMatAll = featMatAll.drop(columns=\
                             list(featMatAll.columns[featMatAll.isna().sum()>featMatAll.shape[0]/10]))

#zscore
def z_score(df):
    return (df-df.mean())/df.std(ddof=0)

featuresZ = featMatAll.select_dtypes(include=['float64']).drop(columns = 'concentration')
descriptor = featMatAll.select_dtypes(include = object)
descriptor = pd.concat([descriptor,featMatAll['concentration'], featMatAll['date']],axis=1)
featMatZ = featuresZ.apply(z_score)
featMatZ = featMatZ.fillna(featMatZ.mean(axis=0))

drugs = descriptor['drug']
concs = descriptor ['concentration']
dates= descriptor['date']

allDrugs = np.unique(drugs)
allConcs = np.unique(concs)
allDates = np.unique(dates)

featMatZ2 = pd.concat([featMatZ, drugs, concs, dates], axis=1)


#%%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#make array of z-scored data
X = np.array(featMatZ.values)
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
PC_df['date'] = dates

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

plt.xlim (-1, 1)
plt.ylim (-1, 1)
plt.xlabel('%' + 'PC_1 (%.2f)' % (pca.explained_variance_ratio_[0]*100), fontsize = 16)
plt.ylabel('%' + 'PC_2 (%.2f)' % (pca.explained_variance_ratio_[1]*100), fontsize = 16)
plt.show()
#plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'agar_biplot.png'))

#make the PCA plots
%cd ~/Documents/GitHub/pythonScripts/Functions/
import PCA_analysis as PC_custom
import make_colormaps as mkc
from matplotlib.colors import LinearSegmentedColormap
import PC_traj as PCJ

#reset figure settings
sns.set()

#graded colormap
cmapGraded = mkc.get_colormaps(RGBsteps =7, thirdColorSteps = 8, n_bins=20)

lutGraded = dict(zip(allDrugs, cmapGraded))
for drug in lutGraded:
    plt.register_cmap(name = drug, cmap = lutGraded[drug])

mkc.plot_color_gradients(lutGraded.values(), lutGraded.keys())

savedir = '/Volumes/behavgenom$/Ida/Data/Antipsychotics/Combined_analysis/PestFigures'
PC_custom.PC12_plots(PC_df, [],[], cmap1, savedir, 'tif', 'concentration')
PCmean, PCsem = PC_custom.PC_av(PC_df, [], 'concentration')

PCJ.PC_trajGraded(PCmean, PCsem, [], savedir, '.png', 'concentration', start_end = False,\
                  cum_var = cumvar, legend = 'off')

    