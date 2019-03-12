#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:22:17 2019

@author: ibarlow
"""

""" Script to analyse the N worms dataset that has been split up into 5 minute
chunks. 
1. Are they consistent across the time chunks, and how does it compare to the
15 minute data?

2. What does the PCA look like comparing the 15min and 5 min data?"""

import os   
import sys
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools
from sklearn.metrics.pairwise import euclidean_distances

#add path of custom functions
sys.path.insert(0, '/Users/ibarlow/Documents/GitHub/pythonScripts/Functions')

FoldIn = '/Volumes/behavgenom$/Ida/Data/NwormTests'
control = 'DMSO'
threshold = 0.5

saveDir = os.path.join(FoldIn, 'Figures5Mins')
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

#import features, filenames and metadata
Metadata = pd.DataFrame()
FeatMat = pd.DataFrame()
FilenameMat = pd.DataFrame()
for (root,dirs,files) in os.walk(FoldIn, topdown=True): #topdown = True as depth option? or use glob function
    for f in files:
        wSearch = re.search('window', f)
        if wSearch:
            window= f[wSearch.start():]
            window = int(re.search(r"\d+", window).group())
            if f.startswith('features_summary_tierpsy'):
                FeatFrame = pd.read_csv(os.path.join(root, f), index_col =False)
                FeatFrame['window'] = window
                FeatMat= FeatMat.append(FeatFrame, sort=True).reset_index(drop=True)
                del FeatFrame
            elif f.startswith('filenames_summary_tierpsy'):
                FileFrame = pd.read_csv(os.path.join(root,f), index_col = False)
                FileFrame ['window'] = window
                FilenameMat = FilenameMat.append(FileFrame, sort=True).reset_index(drop=True)
                del FileFrame
        del wSearch
        if f.startswith('metadata.csv'):
            Metadata=Metadata.append(pd.read_csv(os.path.join(root,f), \
                                                 index_col = False)).reset_index(drop=True)
    del root, dirs, files

#adapt the metadata to match up for the multiple time windows
windows =  np.concatenate([np.ones(Metadata.shape[0])*i for i in range(0,3)])
Metadata['drug concentration'] = Metadata['drug concentration'].astype(int)
Metadata['basename'] =   Metadata['filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
Metadata = pd.concat([Metadata]*np.unique(FeatMat.window).shape[0])
Metadata['uniqueID'] = list(zip(Metadata.basename, windows.astype(int)))

FilenameMat['basename'] = FilenameMat['file_name'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:-1]))
FilenameMat['uniqueID'] = list(zip(FilenameMat.basename, FilenameMat.window))

DrugInfo = pd.concat([FilenameMat.set_index('uniqueID'), Metadata.set_index('uniqueID')], axis=1,sort=True)
DrugInfo['windowID'] = list(zip(DrugInfo.file_id, DrugInfo.window))

FeatMat['windowID'] = list(zip(FeatMat.file_id, FeatMat.window))

FeatMatFinal = pd.concat([FeatMat.set_index('windowID'), DrugInfo.set_index('windowID')[['drug type', \
                          'drug concentration', 'worm number', 'date (YEARMODA)', \
                          'is_good']]], axis=1, sort=True)
    
#drop all the Haloperidol data
FeatMatFinal = FeatMatFinal[FeatMatFinal['drug type']!='Haloperidol']

#filter out features with too many nans and bad files
#1. filter
BadFiles = np.where(FeatMatFinal.isna().sum(axis=1)>FeatMatFinal.shape[1]*threshold)[0]#put threshold at top
BadFiles = np.unique(np.append(BadFiles, np.where(FeatMatFinal['is_good']==False)[0]))
FeatMatFinal = FeatMatFinal.drop(columns=FeatMatFinal.columns[FeatMatFinal.isna().sum(axis=0)>FeatMatFinal.shape[0]*threshold])
FeatMatFinal = FeatMatFinal.reset_index(drop=True).drop(index=BadFiles)
DrugInfo = DrugInfo.reset_index(drop=True).loc[FeatMatFinal.index]

