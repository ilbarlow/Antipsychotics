#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 09:40:26 2019

@author: ibarlow

Script to analyse the 1, 5, 10 worm experiments to determine:
    1) Is there day to day variation in observed drug effects?
    
    2) Do the observed effects depend on the number of worms per plate?
        - are the features consistent?
        - what margin of error can be afforded in counting worms per plate?"""

import os   
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools
from sklearn.metrics.pairwise import euclidean_distances

#add path of custom functions
sys.path.insert(0, '/Users/ibarlow/Documents/GitHub/pythonScripts/Functions')

#custom functions
import DrugFeats
from statsFeats import statsFeat

#set input
FoldIn = '/Volumes/behavgenom$/Ida/Data/NwormTests'
#set control and threshold
control = 'DMSO'
threshold = 0.5

#make a list of selected features to look at
testfeatures = ['angular_velocity_abs_90th', 'relative_to_body_angular_velocity_hips_abs_50th',\
                'blob_quirkiness_10th','major_axis_norm_IQR']

#set folder for saving figures
saveDir = os.path.join(FoldIn, 'Figures')
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

#define function to do statistics and multiple comparisons

    
#import features, filenames and metadata
Metadata = pd.DataFrame()
for (root,dirs,files) in os.walk(FoldIn): #topdown = True as depth option? or use glob function
    for f in files:
        if f.startswith('features_summary_tierpsy'):
            FeatMat = pd.read_csv(os.path.join(root, f), index_col =False)
        if f.startswith('filenames_summary_tierpsy'):
            FilenameMat = pd.read_csv(os.path.join(root,f), index_col = False)
        if f.startswith('metadata.csv'):
            Metadata=Metadata.append(pd.read_csv(os.path.join(root,f), \
                                                 index_col = False))
    del root, dirs, files

#join dataframes
Metadata = Metadata.reset_index(drop = True)
Metadata['drug concentration'] = Metadata['drug concentration'].astype(int)
Metadata['basename'] =   Metadata['filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
FilenameMat['basename'] = FilenameMat['file_name'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:-1]))

#match up features, filenames with the correct date, drug, and N worms from metadata file 
DrugInfo = pd.concat([FilenameMat.set_index('basename'), Metadata.set_index('basename')], axis=1,sort=True)

#sort both dataframes by file id, which is set as index and then reset to fill gaps
DrugInfo = DrugInfo.sort_values(by=['file_id']).set_index('file_id')
FeatMat = FeatMat.sort_values(by=['file_id']).set_index('file_id')

#combine to make big dataFrame with all features and metadata
FeatMatFinal = pd.concat([FeatMat, DrugInfo[['drug type', 'drug concentration', \
                                             'worm number', 'date (YEARMODA)',\
                                             'is_good']]], axis=1)

#drop all the Haloperidol data
FeatMatFinal = FeatMatFinal[FeatMatFinal['drug type']!='Haloperidol']

# =============================================================================
# PROCESSING
# 1. Filter out features with too many nans and rows with too many nans 
# 2. Normalise data by z scoring by day and entire dataset
# 
# EDA
# 3. Select some features and compare the 1,5 and 10 worm data for each day
# 4. Euclidean distance calculation
#
# STATISTICAL COMPARISONS
# 4. does the number of worms affect how many features are significantly different?
# 5. are these features consistent across dates and worms (pairwise)?
# 6. is there a core set of features?
# =============================================================================

#1. filter
BadFiles = np.where(FeatMatFinal.isna().sum(axis=1)>FeatMatFinal.shape[1]*threshold)[0]#put threshold at top
BadFiles = np.unique(np.append(BadFiles, np.where(FeatMatFinal['is_good']==False)[0]))
FeatMatFinal = FeatMatFinal.drop(columns=FeatMatFinal.columns[FeatMatFinal.isna().sum(axis=0)>FeatMatFinal.shape[0]*threshold])
FeatMatFinal = FeatMatFinal.drop(index=BadFiles)
DrugInfo = DrugInfo.loc[FeatMatFinal.index]

    #multilevel index
indexArray = [DrugInfo['date (YEARMODA)'], DrugInfo['worm number'], \
                       DrugInfo['drug type']]
indexApply = pd.MultiIndex.from_arrays(indexArray, names=('date', \
                                                          'worm_number', \
                                                          'drug'))
FeatMatFinal.index = indexApply

    #remove the corresponding  columns from FeatMatFinal
FeatMatFinal = FeatMatFinal.drop(columns = ['date (YEARMODA)', 'worm number', \
                                            'drug type', 'drug concentration', 'is_good'])

#fill any other nans
FeatMatFinal = FeatMatFinal.fillna(FeatMatFinal.mean(axis=0))

#2. Normalise data
    # by day
datesAll = list(FeatMatFinal.index.levels[0])
nworms = list(FeatMatFinal.index.levels[1])
drugsAll = list(FeatMatFinal.index.levels[2])
    #all data
FeatMatZ = pd.DataFrame(stats.zscore(FeatMatFinal, ddof = 1,axis=0), columns = FeatMatFinal.columns, index=FeatMatFinal.index)

#3. Plot selected features
    # group data by worm number and plot each worm number for each feature by date collected
groupedFeatMatWorms = FeatMatZ.groupby('worm_number')
for n in nworms:
    for f in testfeatures:
        g= sns.FacetGrid(groupedFeatMatWorms.get_group(n).reset_index([0,2]), col = 'date')
        g.map(sns.swarmplot, 'drug', f, order = drugsAll)
        g.set_xticklabels(rotation=20, ha='center')    
        plt.savefig(os.path.join(saveDir, '{}worms_{}_swarms.tif'.format(n, f)),\
                    bbox_inches='tight', pad_inches=0.03)
    plt.close('all')    
del f,n

#4. calculate the euclidean distance from DMSO controls for each day and compare across days (take average for each day and then calculate)
groupedFeatMatWorms = FeatMatZ.groupby(['worm_number', 'drug', 'date'])
EucDist= pd.DataFrame()
for n in nworms:
    for drug in drugsAll:
        for date in datesAll:
            try:
                EucDist = EucDist.append(pd.Series({'eDist': euclidean_distances(np.array([groupedFeatMatWorms.get_group((n,drug,date)).mean(axis=0).values,\
                                    groupedFeatMatWorms.get_group((n,'DMSO',date)).mean(axis=0).values]))[0,1],\
                                                    'drug': drug,\
                                                    'date': str(date),\
                                                    'nworms':n}),ignore_index=True)
            except KeyError:
                print('no data for {} worms on {} drug on {}'.format(n, drug, date))
    
#EucDist = pd.DataFrame(distance.squareform(distance.pdist(FeatMatZ, metric= 'euclidean')),\
#                       index=FeatMatZ.index) #scikit pairwise distance is more efficient function
#
#    #need to reset the index to find the corresponding edists
#EucDist = EucDist.reset_index([0,1,2])
#EucDistGrouped = EucDist.groupby(['worm_number'])# group by worm number
#
#    #make eDist DataFrame
#eDistDF = pd.DataFrame()
#for n in nworms:
#    plotDF = pd.concat([EucDistGrouped.get_group(n)[list(EucDistGrouped.get_group((n))[EucDistGrouped.get_group(n)['drug']=='DMSO'].index)],\
#                                 EucDistGrouped.get_group((n))[['drug', 'date']]], axis=1)
#    plotDF['edist'] = plotDF.select_dtypes(include='float').mean(axis=1) #make sure this excludes concentratin and n - refactor?
#    plotDF['nworms'] = n  
#    eDistDF = eDistDF.append(plotDF[['drug', 'date', 'edist', 'nworms']])
#    del plotDF
#eDistDF = eDistDF.reset_index(drop=True)    

    #plot combined data
g = sns.FacetGrid(EucDist, col = 'nworms', hue = 'date')
g.map (sns.swarmplot, 'drug', 'eDist', order = drugsAll).add_legend()
g.set_xticklabels(rotation=20, ha='right')    
plt.savefig(os.path.join(saveDir, 'Edist_swarms.tif'), bbox_inches='tight',\
            pad_inches=0.03)    
plt.close('all')
 
#4. What features are statistically different - separate by date

pValst_testDF, bh_ttDF = statsFeat(FeatMatFinal, 'normal', 0.1, control)
pValsRankSumsDF, bh_rankDF = statsFeat(FeatMatFinal, 'notnormal', 0.1, control)

    #plot total number of significant features by worm number, drug and day
sns.catplot(x = 'worm_number', y = 'sumSig', data = bh_ttDF, hue = 'date', \
            col = 'drug', kind ='bar', palette = 'colorblind')
plt.savefig(os.path.join(saveDir, 'T_test_number_sig_feats.tif'), bbox_inches='tight', \
            pad_inches=0.03)
    #for rank sum test
sns.catplot(x = 'worm_number', y = 'sumSig', data = bh_rankDF, hue = 'date', \
            col = 'drug', kind ='bar', palette = 'colorblind')
plt.savefig(os.path.join(saveDir, 'rankSumtest_number_sig_feats.tif'), bbox_inches='tight', \
            pad_inches=0.03)

#5. Which features are consistent across worm numbers and dates
    #create drugFeat object
bhP_grouped = bh_rankDF.groupby(['date', 'drug', 'worm_number'])
sigFeats = {}
for drug in drugsAll:
    if drug != control:
        sigFeats[drug]=[] #class object
        for date in datesAll:
            for n in nworms:
                try:
                    sigFeats[drug].append(DrugFeats.DrugFeats(bhP_grouped.get_group((date,drug,n))))
                except KeyError:
                    print ('no data for {} {} drug with {} worms'.format(date,drug,n))
            

   # make a dataframe of feature lists that are consistent across number of worms and dates
   #this kind of copies the above so is a bit redundant
consistentFeatsDF = pd.DataFrame()
for drug in sigFeats:
    if drug !=control:
        for p,r in itertools.combinations(sigFeats[drug],2): #pairwise combinations
            if p.date == r.date:
                consistentFeatsDF = consistentFeatsDF.append(pd.Series({'drug': p.drug, \
                           'comparison': '{} vs {} worms'.format(p.nworms, r.nworms),\
                           'features':list(set(p.FeatureList).intersection(r.FeatureList)),\
                           'nFeatures': len(list(set(p.FeatureList).intersection(r.FeatureList))),\
                           'date': str(p.date),\
                           'nworms': 'n/a'}), ignore_index=True)
            elif p.nworms == r.nworms:
                consistentFeatsDF = consistentFeatsDF.append(pd.Series({'drug': p.drug, \
                           'comparison': '{} vs {}'.format(p.date, r.date),\
                           'features':list(set(p.FeatureList).intersection(r.FeatureList)),\
                           'nFeatures': len(list(set(p.FeatureList).intersection(r.FeatureList))),\
                           'date': 'n/a',\
                           'nworms': p.nworms}), ignore_index=True)
    
    #some plots - general 
consistentFeatsGrouped = consistentFeatsDF.groupby('comparison')
compsAll = np.unique(consistentFeatsDF['comparison'])
for c in compsAll:
    if 'worms' in c:
        plt.figure()
        sns.swarmplot('drug', 'nFeatures', \
                      data=consistentFeatsGrouped.get_group(c), hue='date')
        plt.title(c)
    if '20' in c:
        plt.figure()
        sns.swarmplot('drug', 'nFeatures',\
                      data=consistentFeatsGrouped.get_group(c), hue='nworms')
        plt.title(c)
        
    #worm comparisons
g = sns.FacetGrid(consistentFeatsDF[consistentFeatsDF['date']!='n/a'], \
                  col = 'drug', hue='date', sharey=False)
g.map(sns.swarmplot,'comparison', 'nFeatures', \
      order = [(i) for i in compsAll if 'worms' in i]).add_legend()
g.set_xticklabels(rotation=20, ha='right')
plt.show()
plt.savefig(os.path.join(saveDir, 'ranksum_nFeatures_consistent_nworms.tif'), bbox_inches='tight', \
            pad_inches=0.03)

    #date comparisons
g = sns.FacetGrid(consistentFeatsDF[consistentFeatsDF['date']=='n/a'], \
                  col = 'drug', hue='nworms',sharey=False)
g.map(sns.swarmplot,'comparison', 'nFeatures',\
      order = [(i) for i in compsAll if '20' in i]).add_legend()
g.set_xticklabels(rotation=20, ha='right')
plt.show()
plt.savefig(os.path.join(saveDir, 'rank_sum_nFeatures_consistent_dates.tif'), bbox_inches='tight', \
            pad_inches=0.03)

#6. define list of core features
    #for each number of worms find the core features that are consistently changed
coreFeatures = pd.DataFrame()
for drug in drugsAll:
    if drug != control: # != control
        for n in nworms:
            flist = [set(sigFeats[drug][i].FeatureList) for i in range(0,len(sigFeats[drug])) if sigFeats[drug][i].nworms ==n]
            flist = list(flist[0].intersection(*flist[1:]))
            coreFeatures = coreFeatures.append(pd.Series\
                                              ({'Features': flist,\
                                                'drug': drug,\
                                                'nworms': n,\
                                                'nFeatures': len(flist)}), ignore_index=True)
            
    #make figure
plt.figure()
sns.lineplot(x = 'nworms', y = 'nFeatures', data =coreFeatures, hue = 'drug')
plt.savefig(os.path.join(saveDir, 'rankSum_nCoreFeatures.tif'), bbox_inches='tight', \
            pad_inches=0.03)

# =============================================================================
# Still to do
# consistent features across dates

#find strings (as in head angular velocity) that match of consistent features across day
# ============================================================================
