#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 10:11:30 2018

@author: ibarlow
"""

"""ICA of antipsychotics"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
del dirs,root,files, file1

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
featMatAll =featMatAll.drop(columns = list(featMatAll.columns[featMatAll.std()==0.0]))

featMatAll= featMatAll.fillna(featMatAll.mean(axis=0))                            

#get descriptors
featuresZ = featMatAll.select_dtypes(include=['float64']).drop(columns = 'concentration')
descriptor = featMatAll.select_dtypes(include = object)
descriptor = pd.concat([descriptor, featMatAll['concentration'], featMatAll['date']],axis=1)
drugs = descriptor['drug']
concs = descriptor ['concentration']
dates= descriptor['date']
allDrugs = np.unique(drugs)
allConcs = np.unique(concs)
allDates = np.unique(dates)

#zscore
def z_score(df):
    return (df-df.mean())/df.std(ddof=0)

featMatZ = featuresZ.apply(z_score)
featMatZ = featMatZ.fillna(featMatZ.mean(axis=0))

featMatZ2 = pd.concat([featMatZ, drugs, concs, dates], axis=1)

#%% now apply ICA to the new and old datasets separately and then combine
from sklearn.decomposition import FastICA

DataGroups = {'old':[12042017, 9062017, 7062017],\
              'new': [20180906, 20181005, 20181011, 20181026]}

grouped = featMatZ2.groupby('date')
oldFeatMatZ = pd.DataFrame()
for item in DataGroups['old']:
    oldFeatMatZ = oldFeatMatZ.append(grouped.get_group(item))
oldFeatMatZ = oldFeatMatZ.reset_index(drop=True)

newFeatMatZ = pd.DataFrame()
for item in DataGroups['new']:
    newFeatMatZ = newFeatMatZ.append(grouped.get_group(item))
    

#just on old dataset to start with
X = oldFeatMatZ.select_dtypes(include= 'float').drop(columns= 'concentration')
transformer = FastICA(max_iter=10000)
Xt = transformer.fit_transform(X)

ICnames = ['IC_%d' %(i+1) for i in range(0, Xt.shape[1])]
IC_df = pd.DataFrame(data=Xt, columns = ICnames)
IC_df['drug'] = oldFeatMatZ['drug']
IC_df['concentration'] = oldFeatMatZ['concentration']
IC_df['date']= oldFeatMatZ['date']

import make_colormaps as mkc
from matplotlib.colors import LinearSegmentedColormap
import PC_traj as PCJ
import PCA_analysis as PC_custom


cmap1 = sns.color_palette('tab20',len(np.unique(drugs)))
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


#make the PC plots
savedir =  '/Volumes/behavgenom$/Ida/Data/Antipsychotics'
PC_custom.PC12_plots(IC_df, [],[], cmap1, savedir, 'tif', 'concentration')
ICmean, ICsem = PC_custom.PC_av(IC_df, [], 'concentration')

PCJ.PC_trajGraded(ICmean, ICsem,['IC_1','IC_2'], [], savedir, '.png', 'concentration', start_end = False,\
                  cum_var = None, legend = 'off')

#find the features that contribute most to the ICA
