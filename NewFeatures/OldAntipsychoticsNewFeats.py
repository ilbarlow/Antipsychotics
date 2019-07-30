#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:52:28 2018

@author: ibarlow
"""

""" Scipt to combine features from old Antipsychotics experiments and new ones"""
%cd /Documents/GitHub/pythonscripts/Functions

import os
import pandas as pd
import displayfiles as disp
from scipy import stats

dirFeats, FeatFiles = disp.displayfiles('featuresN.hdf5', inputfolder = None, outputfile = 'Featurefiles.tsv')

#load these features files
features = pd.DataFrame()
for item in FeatFiles:
    with pd.HDFStore(item, 'r') as fid:
        temp = fid['/features_stats'].pivot_table(columns = 'name', values = 'value')
        temp ['exp'] = os.path.basename(item)
        features = features.append(temp)
    del temp
features = features.reset_index(drop=True)

#remove features where all rows have the same value
to_remove = list(features.columns[features.std()==0])
features = features.drop(columns = to_remove)

#now match up metadata with the files and add in drug, concentration information
    #can assume that both FeatFiles and metadata are sorted by set and channel
features2 = pd.DataFrame()
for i in range(0,features.shape[0]):
    update = features.iloc[i,:]
    exp = features.iloc[i]['exp'].split('_')
    if len(exp) == 10:
        update['drug'] = exp[2]
        update['concentration'] = float(exp[3])
        update ['date'] = exp[-3]
    else:
        update['drug'] = '_'.join(exp[2:4])
        update['concentration'] = float(exp[4])
        update['date'] = exp[-3]
    
    features2 = features2.append(update)
    
    del update,exp

features2 = features2.reset_index(drop=True)


##save the dataframe to a csv
features2.to_csv(os.path.join(os.path.dirname(dirFeats), 'FeatMatAllNewFeats.csv'), index=False)


#Zscoring
#drop exp for zscoring 
featuresZ = features.drop(columns=['exp'])
featuresZ = pd.DataFrame(stats.zscore(featuresZ, axis=0), columns = featuresZ.columns)

metadata = pd.read_csv(os.path.join(os.path.dirname(dirFeats), 'metadata.csv' ))
