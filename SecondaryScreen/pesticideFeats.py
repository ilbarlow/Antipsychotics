#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:38:50 2018

@author: ibarlow
"""

""" Load pesticides data so that it is in the same format as the antipsychotics 
and can be combined"""

import pandas as pd
import os
import numpy as np

pestFeats = '/Volumes/behavgenom$/Eleni/pesticides/Results-new_features/features_summary_tierpsy_plate_20180829_101747.csv'
pestNames = '/Volumes/behavgenom$/Eleni/pesticides/Results-new_features/filenames_summary_tierpsy_plate_20180829_101747.csv'
pestMoA = '/Volumes/behavgenom$/Ida/Data/Antipsychotics/Pesticides/Initial_Pesticide_Screen Drugs.csv'

pestFeatMat = pd.read_csv(pestFeats, index_col= False)
pestNameMat = pd.read_csv(pestNames,index_col=False)
pestMoAMat = pd.read_csv(pestMoA, index_col = False)
pestMoAMat = pestMoAMat.fillna(np.nan)

#split the filenames to extract out Drug, concentration, date
pestNameMat['drug'] = [i[1].file_name.split('_')[7] for i in pestNameMat.iterrows()]
pestNameMat['date'] = [i[1].file_name.split('_')[-3] for i in pestNameMat.iterrows()]

#different for concentrations
concs = []
for i in pestNameMat.iterrows():
    try:
        concs.append(float(i[1].file_name.split('_')[8]))
    except ValueError:
        concs.append(float(0))

pestNameMat['concentration'] = concs
#replace no with no_compound
pestNameMat.replace(to_replace = 'No', value = 'No_Compound', inplace=True)


if (pestNameMat['file_id'] == pestFeatMat['file_id']).sum() == pestFeatMat.shape[0]:
    pestFeatFinal = pd.concat([pestFeatMat, pestNameMat[['drug', 'concentration', 'date']]], axis=1)
    pestFeatFinal = pestFeatFinal.drop(columns = 'file_id')
else:
    print ('fileID mismatch')

#add in the MoA
allDrugs = np.unique(pestNameMat['drug'])
pestFeatFinal2 = pd.DataFrame()
for drug in allDrugs:
    try: 
        deets = pestMoAMat[pestMoAMat['CSN'] == drug]
        temp = pestFeatFinal[pestFeatFinal['drug'] == drug].copy()
        temp.loc[:,'MoAGeneral'] = str(deets['MOA general'].values[0])
        temp.loc[:,'MoASpecific'] = str(deets['MOA specific'].values[0])
    
    except IndexError:
        temp.loc[:,'MoAGeneral'] = np.nan
        temp.loc[:,'MoAGeneral'] = np.nan
        
    pestFeatFinal2 = pestFeatFinal2.append(temp, sort=True)
    
    del temp, deets

pestFeatFinal2 = pestFeatFinal2.reset_index(drop=True)
    
saveDir = '/Volumes/behavgenom$/Ida/Data/Antipsychotics/Pesticides/AuxiliaryFiles'

pestFeatFinal2.to_csv(os.path.join(saveDir, 'FeatMatAllNewFeats.csv'), index=False)


#%% Old antipsychotics features

oldAntiFeat = '/Volumes/behavgenom$/Eleni/antipsychotics/Results-new features/Agar/features_summary_tierpsy_plate_20180830_190509.csv'
oldAntiNames = '/Volumes/behavgenom$/Eleni/antipsychotics/Results-new features/Agar/filenames_summary_tierpsy_plate_20180830_190509.csv'

oldAntiFeatMat = pd.read_csv(oldAntiFeat, index_col= False)
oldAntiNameMat = pd.read_csv(oldAntiNames,index_col=False)

#split the filenames to extract out Drug, concentration, date
oldAntiNameMat['drug'] = [i[1].file_name.split('_')[5] for i in oldAntiNameMat.iterrows()]
oldAntiNameMat['date'] = [i[1].file_name.split('_')[-3] for i in oldAntiNameMat.iterrows()]

#different for concentrations
concs = []
for i in oldAntiNameMat.iterrows():
    try:
        concs.append(float(i[1].file_name.split('_')[6]))
    except ValueError:
        concs.append(float(0))

oldAntiNameMat['concentration'] = concs
oldAntiNameMat.replace(to_replace='No', value = 'No_Compound', inplace=True)

if (oldAntiNameMat['file_id'] == oldAntiFeatMat['file_id']).sum() == oldAntiFeatMat.shape[0]:
    oldAntiFeatFinal = pd.concat([oldAntiFeatMat, oldAntiNameMat[['drug', 'concentration', 'date']]], axis=1)
    oldAntiFeatFinal = oldAntiFeatFinal.drop(columns = 'file_id')
else:
    print ('fileID mismatch')

saveDir = '/Volumes/behavgenom$/Ida/Data/Antipsychotics/OriginalDataSet/AuxiliaryFiles'

oldAntiFeatFinal.to_csv(os.path.join(saveDir, 'FeatMatAllNewFeats.csv'), index=False)
