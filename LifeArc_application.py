#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:51:20 2019

@author: ibarlow
"""

""" script to generate screen stats and Z-factor numbers for LifeArc

Use Antipsychotics data to find the average and standard deviation for High Control
(Drug with stong effect eg. Chlorpromazine HCl) and the background (DMSO)

Use these data to calculate the Z-factor as well, as determined in 
Zhang et al 1999 (J. Biomol screen. 4, 67-73)"""

import pandas as pd
import numpy as np
import os
from scipy import stats


feat_file = '/Volumes/behavgenom$/Ida/Data/Antipsychotics/features_summary_tierpsy_plate_20190531_162311.csv'
filename_file = '/Volumes/behavgenom$/Ida/Data/Antipsychotics/filenames_summary_tierpsy_plate_20190531_162311.csv'
metadata_file = '/Volumes/behavgenom$/Ida/Data/Antipsychotics/metadata_all.csv'

background = 'DMSO'
HighControl = 'Chlopromazine hydrocholoride'
testFeature1 = 'relative_to_body_speed_midbody_IQR'
testFeature2 = 'eigen_projection_2_abs_IQR'

#import data
featMat = pd.read_csv(feat_file, index_col='file_id')
filenameMat = pd.read_csv(filename_file, index_col='file_id')
metadata = pd.read_csv(metadata_file, index_col= False)

#make big dataframe with all the data
featMat_all = pd.concat([featMat, filenameMat], axis=1, join='inner')
featMat_all['basename'] = featMat_all['file_name'].apply(lambda x: '_'.join(os.path.basename(x).split('_')[:-1]))

#get basename of metadata too
metadata['basename'] = metadata['filename'].apply(lambda x: '_'.join(os.path.basename(x).split('.')[:-1]))

#concat
featMat_metadata= pd.concat([featMat_all.set_index('basename'),
                             metadata.set_index('basename')], axis=1, join='inner')
featMatFinal = featMat_metadata.reset_index(drop=False)
#drop bad files
featMatFinal = featMatFinal[featMatFinal.is_good==True]
featMatFinal.drop(columns = ['is_good', 'file_name'], inplace=True)

#zscore data before calculating z-factor
featMatFinal.fillna(featMatFinal.mean(axis=0), inplace=True)
#stats.zscore
FeatMatZ = pd.DataFrame(stats.zscore(featMatFinal.drop(columns = metadata.columns),
                        ddof= 1,
                        axis=0),
                        columns = featMatFinal.drop(columns =metadata.columns).columns)
FeatMatZ = pd.concat([FeatMatZ, metadata],axis=1)

#now extract out feature values
featMatFinal_grouped = featMatFinal.groupby('drug type')

#example code to run to get the values
featMatFinal_grouped.get_group(background)[testFeature1].mean()
featMatFinal_grouped.get_group(HighControl)[testFeature1].mean()

featMatFinal_grouped.get_group(background)[testFeature1].std()
featMatFinal_grouped.get_group(HighControl)[testFeature1].std()

featMatFinal_grouped.get_group(background)[testFeature2].mean()
featMatFinal_grouped.get_group(HighControl)[testFeature2].mean()

featMatFinal_grouped.get_group(background)[testFeature2].std()
featMatFinal_grouped.get_group(HighControl)[testFeature2].std()


#calculate Z-factor
