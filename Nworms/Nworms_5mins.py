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
import glob  
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
from statsFeats import statsFeat

def find_window(filename):
    """ Little function to find the window number 
    Input - filename
    
    Output - the window as int
    """
    if re.search(r"_window_\d+", filename):
        window = int(re.search(r"\d+", re.search(r"_window_\d+", filename).group()).group())
    return window


if __name__ == '__main__':

    FoldIn = '/Volumes/behavgenom$/Ida/Data/NwormTests'
    control = 'DMSO'
    threshold = 0.5
    
    save_dir = os.path.join(FoldIn, 'Figures5Mins')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #import features, filenames and metadata
    feat_files = glob.glob(os.path.join(FoldIn, 'features_summary_tierpsy*'))
    filename_files = glob.glob(os.path.join(FoldIn, 'filenames_summary_tierpsy*'))
    meta_files = glob.glob(os.path.join(FoldIn, '**/*metadata.csv'), recursive=True)
    
    Metadata = pd.DataFrame()
    FeatMat = pd.DataFrame()
    FilenameMat = pd.DataFrame()
    for f in feat_files:
        try:
            _window = find_window(f)
        except Exception:
            print ('{} has no window data'.format(f))
            continue
        _FeatFrame = pd.read_csv(f, index_col =False)
        _FeatFrame['window'] = _window
        FeatMat= FeatMat.append(_FeatFrame, sort=True).reset_index(drop=True)
    
    for fn in filename_files:
        try:
            _window = find_window(fn)
        except Exception:
            print ('{} has no window data'.format(fn))
            continue
        _FileFrame = pd.read_csv(fn, index_col = False)
        _FileFrame ['window'] = _window
        FilenameMat = FilenameMat.append(_FileFrame, sort=True).reset_index(drop=True)
        
    for m in meta_files:
         Metadata=Metadata.append(pd.read_csv(m, index_col = False)).reset_index(drop=True)
         
    #adapt the metadata to match up for the multiple time windows
    _windows =  np.concatenate([np.ones(Metadata.shape[0])*i for i in range(0,FeatMat['window'].max()+1)])
    Metadata['drug concentration'] = Metadata['drug concentration'].astype(int)
    Metadata['basename'] =   Metadata['filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
    Metadata = pd.concat([Metadata]*np.unique(FeatMat.window).shape[0])
    Metadata['uniqueID'] = list(zip(Metadata.basename, _windows.astype(int)))
    
    FilenameMat['basename'] = FilenameMat['file_name'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:-1]))
    FilenameMat['uniqueID'] = list(zip(FilenameMat.basename, FilenameMat.window))
    
    #concat the metadata
    DrugInfo = pd.concat([FilenameMat.set_index('uniqueID'), Metadata.set_index('uniqueID')], axis=1,sort=True)
    DrugInfo['windowID'] = list(zip(DrugInfo.file_id, DrugInfo.window))  
    FeatMat['windowID'] = list(zip(FeatMat.file_id, FeatMat.window))
    
    #drop all the Haloperidol data
    DrugInfo= DrugInfo[DrugInfo['drug type'] != 'Haloperidol']
    
    FeatMatFinal = pd.concat([FeatMat.set_index('windowID'), DrugInfo.set_index('windowID')[['drug type', \
                              'drug concentration', 'worm number', 'date (YEARMODA)', \
                              'is_good']]], join='inner', axis=1, sort=True)# join = inner to joing intersection of indices
    
    #filter out features with too many nans and bad files
    #1. filter
    BadFiles = np.where(FeatMatFinal.isna().sum(axis=1)>FeatMatFinal.shape[1]*threshold)[0]
    BadFiles = np.unique(np.append(BadFiles, np.where(FeatMatFinal['is_good']==False)[0]))
    FeatMatFinal = FeatMatFinal.drop(columns=FeatMatFinal.columns[FeatMatFinal.isna().sum(axis=0)>FeatMatFinal.shape[0]*threshold])
    FeatMatFinal = FeatMatFinal.reset_index(drop=True).drop(index=BadFiles)
    DrugInfo = DrugInfo.reset_index(drop=True).loc[FeatMatFinal.index]
    
    #apply multilevel index to FeatMatFinal
    _indexArray = [DrugInfo['date (YEARMODA)'], DrugInfo['worm number'], DrugInfo['drug type'],\
                           DrugInfo['window']]
    _indexApply = pd.MultiIndex.from_arrays(_indexArray, names = ('date',\
                                                                'worm_number',\
                                                                'drug',\
                                                                'window'))
    FeatMatFinal.index = _indexApply
    
    FeatMatFinal = FeatMatFinal.drop(columns = ['date (YEARMODA)', 'drug type', 'drug concentration',\
                                                'window', 'worm number', 'is_good'])
    
    #do stats grouped by window number
    FeatMatGrouped = FeatMatFinal.groupby(['date', 'worm_number', 'drug', 'window'])
    
    # stats    
    pValsDF, bhDF = statsFeat(FeatMatGrouped, 'normal', 0.1, control)
    bhDF= bhDF.reset_index()
    bhDF['drug'] = [list(i) for i in bhDF.metadata.str if i.dtype == object][0]
    bhDF['worm_number'] = bhDF.metadata.str[1]
    bhDF['window'] = bhDF.metadata.str[3]
    bhDF['date'] = bhDF.metadata.str[0]
    
    
    #plot the total number of sig feats by worm number and drug (with window as hue) for each date
    for date in np.unique(bhDF['date']):
        sns.catplot(x = 'worm_number', \
                    y = 'sumSig',\
                    data = bhDF[bhDF['date']==date], \
                    hue = 'window', \
                    col = 'drug',\
                    kind ='bar', \
                    palette = 'colorblind')
#            plt.savefig(os.path.join(save_dir, 'T_test_number_sig_feats.tif'), bbox_inches='tight', \
#            pad_inches=0.03)
    
    # TODO make figure comparing 5 min data to 15 minute data
    
    
    #normalise the data
    FeatMatFinal= FeatMatFinal.fillna(FeatMatFinal.mean(axis=0))
    FeatMatZ = pd.DataFrame(stats.zscore(FeatMatFinal, ddof=1, axis=0), \
                            columns = FeatMatFinal.columns,\
                            index = FeatMatFinal.index)
