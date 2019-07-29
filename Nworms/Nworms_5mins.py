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
    try:
        window = int(re.search(r"\d+", re.search(r"_window_\d+", filename).group()).group())
        return window+1 #add one as window indexing starts at 0
    except Exception as error:
        print ('{} has error:{}'.format(filename, error))
        return


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
    
    Metadata = {'chunked': pd.DataFrame(), 'unchunked':pd.DataFrame()}
    FeatMat={'chunked': pd.DataFrame(), 'unchunked':pd.DataFrame()}
    FilenameMat={'chunked': pd.DataFrame(), 'unchunked':pd.DataFrame()}
    DrugInfo = {'chunked': pd.DataFrame(), 'unchunked':pd.DataFrame()}
    for f in feat_files:
        _window = find_window(f)
        if _window:
            _FeatFrame = pd.read_csv(f, index_col =False)
            _FeatFrame['window'] = _window
            FeatMat['chunked']= FeatMat['chunked'].append(_FeatFrame,\
                   sort=True).reset_index(drop=True)
        else:
            FeatMat['unchunked']= pd.read_csv(f, index_col=False)
            FeatMat['unchunked']['window'] =-1 #set as negative for grouping purposes
        
    
    for fn in filename_files:
        _window = find_window(fn)
        if _window:
            _FileFrame = pd.read_csv(fn, index_col = False)
            _FileFrame ['window'] = _window
            FilenameMat['chunked'] = FilenameMat['chunked'].append(_FileFrame,\
                       sort=True).reset_index(drop=True)
        else:
            FilenameMat['unchunked'] = pd.read_csv(fn, index_col=False)
        
    for m in meta_files:
         Metadata['unchunked']=Metadata['unchunked'].append(\
                 pd.read_csv(m, index_col = False)\
                 ).reset_index(drop=True)
         
    #adapt the metadata to match up for the multiple time windows
    _windows =  np.concatenate([np.ones(Metadata['unchunked'].shape[0])*i for i in range(1,FeatMat['chunked']['window'].max()+1)])
    Metadata['chunked'] = pd.concat([Metadata['unchunked']]*np.unique(FeatMat['chunked'].window).shape[0])
    Metadata['chunked'] = pd.concat([Metadata['unchunked']]*np.unique(FeatMat['chunked'].window).shape[0])
    
    #exctract out basename in Metadata and Filename data
    for block in Metadata:
        Metadata[block]['drug concentration'] = Metadata[block]['drug concentration'].astype(int)
        Metadata[block]['basename'] =   Metadata[block]['filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
        FilenameMat[block]['basename'] = FilenameMat[block]['file_name'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:-1]))

    
    #make identifier column for chunked metadata and filename
    Metadata['chunked']['uniqueID'] = list(zip(Metadata['chunked'].basename, _windows.astype(int)))
    FilenameMat['chunked']['uniqueID'] = list(zip(FilenameMat['chunked'].basename, FilenameMat['chunked'].window))
    
    #concat with the metadata
    DrugInfo['chunked'] = pd.concat([FilenameMat['chunked'].set_index('uniqueID'), \
                                    Metadata['chunked'].set_index('uniqueID')], \
                                     axis=1,sort=True)
    DrugInfo['unchunked'] = pd.concat([FilenameMat['unchunked'].set_index('basename'), \
                                      Metadata['unchunked'].set_index('basename')], \
                                      axis=1, join='inner', sort=True)
    
    #unique window identifier required to combine featmat with druginfo chunked data
    DrugInfo['chunked']['windowID'] = list(zip(DrugInfo['chunked'].file_id,\
            DrugInfo['chunked'].window))  
    FeatMat['chunked']['windowID'] = list(zip(FeatMat['chunked'].file_id, \
           FeatMat['chunked'].window))
    
    #drop all the Haloperidol data
    for block in DrugInfo:
        DrugInfo[block]= DrugInfo[block][DrugInfo[block]['drug type'] != 'Haloperidol']
    
    FeatMatFinal = pd.concat([FeatMat['chunked'].set_index('windowID'), \
                              DrugInfo['chunked'].set_index('windowID')[
                                      ['drug type',\
                                       'drug concentration',\
                                       'worm number',\
                                       'date (YEARMODA)', \
                                       'is_good']\
                                       ]], join='inner', axis=1, sort=True).reset_index(drop=True)# join = inner to joing intersection of indices
    
    FeatMatFinal = FeatMatFinal.append(
        pd.concat([FeatMat['unchunked'].set_index('file_id'),\
                   DrugInfo['unchunked'].set_index('file_id')[
                                     ['drug type',
                                     'drug concentration', \
                                     'worm number', \
                                     'date (YEARMODA)',\
                                     'is_good'
                                     ]]], join='inner', axis=1, sort=True), sort=True)

    FeatMatFinal = FeatMatFinal.reset_index(drop=True)
    #filter out features with too many nans and bad files
    #1. filter
    BadFiles = np.where(FeatMatFinal.isna().sum(axis=1)>FeatMatFinal.shape[1]*threshold)[0]
    BadFiles = np.unique(np.append(BadFiles, np.where(FeatMatFinal['is_good']==False)[0]))
    FeatMatFinal = FeatMatFinal.drop(columns=FeatMatFinal.columns[FeatMatFinal.isna().sum(axis=0)>FeatMatFinal.shape[0]*threshold])
    FeatMatFinal = FeatMatFinal.reset_index(drop=True).drop(index=BadFiles)
#    DrugInfo = DrugInfo.reset_index(drop=True).loc[FeatMatFinal.index]
    
    #apply multilevel index to FeatMatFinal
    _indexArray = [FeatMatFinal['date (YEARMODA)'], FeatMatFinal['worm number'], \
                                FeatMatFinal['drug type'],\
                           FeatMatFinal['window']]
    _indexApply = pd.MultiIndex.from_arrays(_indexArray, names = ('date',\
                                                                'worm_number',\
                                                                'drug',\
                                                                'window'))
    FeatMatFinal.index = _indexApply  
    FeatMatFinal = FeatMatFinal.drop(columns = ['date (YEARMODA)',\
                                                'drug type',\
                                                'drug concentration',\
                                                'window', \
                                                'worm number', \
                                                'is_good'\
                                                ])

    #metadata dictionary
    metadata_dict = {}
    for item in FeatMatFinal.index.levels:
        metadata_dict[item.name] = item.values
    
    #do stats grouped by window number
    FeatMatGrouped = FeatMatFinal.groupby(['date', 'worm_number', 'drug', 'window'])
    
    # stats    
    pValsDF, bhDF = statsFeat(FeatMatGrouped, 'normal', 0.1, control)
    bhDF= bhDF.reset_index()
    bhDF['drug'] = [list(i) for i in bhDF.metadata.str if i.dtype == object][0]
    bhDF['date'] = bhDF.metadata.str[0]
    bhDF['worm_number'] = bhDF.metadata.str[1]
    bhDF['window'] = bhDF.metadata.str[-1]
    bhDF =bhDF.drop(columns = 'metadata')
    
    #plot the total number of sig feats by worm number and drug (with window as hue) for each date
    for date in metadata_dict['date']:
        sns.catplot(x = 'worm_number', \
                    y = 'sumSig',\
                    data = bhDF[bhDF['date']==date], \
                    hue = 'window', \
                    col = 'drug',\
                    kind ='bar', \
                    palette = 'colorblind'
                    )
        plt.savefig(os.path.join(save_dir, 'T_test_number_sig_feats5mins_{}.tif'.format(date)), bbox_inches='tight', \
        pad_inches=0.03)
    #and for all data combined
    sns.catplot(x = 'worm_number', \
                y = 'sumSig',\
                data = bhDF, \
                hue = 'window', \
                col = 'drug',\
                kind ='bar', \
                palette = 'colorblind',
                ci= 'sd'
                ) 
    plt.savefig(os.path.join(save_dir,'T_test_number_sigfigs5mins_combined.png'))
    
    #find out what these features are by making a dataframe with sig Feature list,
    #the number of features, and metadata
    bhDF_grouped = bhDF.groupby(['date', 'worm_number', 'drug', 'window'])    
    sig_feature_summary =pd.DataFrame()
    for date in np.unique(bhDF.date):
        for n in np.unique(bhDF.worm_number):
            for drug in np.unique(bhDF.drug):
                for w in np.unique(bhDF.window):
                    try:
                        _feats= pd.Series({'feature_list': list(bhDF.select_dtypes(include='float').columns[bhDF_grouped.get_group((date, n, drug, w)).select_dtypes(include='float').notna().any()]),
                                            'number_features': sum(bhDF_grouped.get_group((date, n, drug, w)).select_dtypes(include='float').notna().sum())})
                        _feats = _feats.append(bhDF_grouped.get_group((date,n,drug,w))[['window', 'drug', 'worm_number', 'date']].reset_index(drop=True).T)
                        
                        sig_feature_summary = sig_feature_summary.append(_feats.T, sort=True)
                    except Exception as error:
                        print (error)
    
    #plot this
    g= sns.catplot(x= 'window',\
                   y= 'number_features',\
                   data = sig_feature_summary,\
                   hue = 'worm_number',\
                   col = 'drug',\
                   kind = 'bar'\
                   )
    
    # TODO string compare the features and find core features with the same root that 
    # are consistent
    
    #go through each of the feature lists broaded into the type of feature - ie remove the summary statistic at the end
    broad_feature_summary = pd.DataFrame()
    for i,r in sig_feature_summary.iterrows():
        r['general_features'] = np.unique(['_'.join(f.split('_')[:-1]) for f in r.feature_list])
        broad_feature_summary = broad_feature_summary.append(r, sort=True)
    
    broad_feature_summary = broad_feature_summary.reset_index(drop=True)
    
    broad_features_grouped = broad_feature_summary.groupby('drug')
    
    consistent_features_df = pd.DataFrame()
    for drug in metadata_dict['drug']:
        if drug !=control:
            for p,r in itertools.combinations(list(r for i,r in broad_features_grouped.get_group(drug).iterrows()), 2): #pairwise combinations
                if p.date == r.date:
                    if p.window != r.window:                        
                        consistent_features_df = consistent_features_df.append(pd.Series({
                                   'drug': p.drug, \
                                   'comparison': '{} vs {} window'.format(p.window, r.window),\
                                   'general_features':list(set(p.general_features).intersection(r.general_features)),\
                                   'nFeatures': len(list(set(p.general_features).intersection(r.general_features))),\
                                   'date': str(p.date),\
                                   'nworms': p.worm_number,\
                                   'window': 'n/a',}),\
                                    ignore_index=True)
                    else:
                        consistent_features_df = consistent_features_df.append(pd.Series({
                                   'drug': p.drug, \
                                   'comparison': '{} vs {} worms'.format(p.worm_number, r.worm_number),\
                                   'general_features':list(set(p.general_features).intersection(r.general_features)),\
                                   'nFeatures': len(list(set(p.general_features).intersection(r.general_features))),\
                                   'date': str(p.date),\
                                   'nworms': p.worm_number,\
                                   'window': p.window,}),\
                                    ignore_index=True)
                        
                elif p.worm_number == r.worm_number:
                    consistent_features_df = consistent_features_df.append(pd.Series({
                               'drug': p.drug, \
                               'comparison': '{} vs {}'.format(p.date, r.date),\
                               'general_features':list(set(p.general_features).intersection(r.general_features)),\
                               'nFeatures': len(list(set(p.general_features).intersection(r.general_features))),\
                               'date': 'n/a',\
                               'nworms': p.worm_number,\
                               'window': '{} vs {} window'.format(p.window, r.window)}),\
                                ignore_index=True)
    
    consistent_features_grouped = consistent_features_df.groupby('comparison')
    compsAll = np.unique(consistent_features_df['comparison'])
    for c in compsAll:
        if 'worms' in c:
            plt.figure()
            sns.swarmplot('drug', 'nFeatures', \
                          data=consistent_features_grouped.get_group(c), hue='date')
            plt.title(c)
        if '20' in c:
            plt.figure()
            sns.swarmplot('drug', 'nFeatures',\
                          data=consistent_features_grouped.get_group(c), hue='nworms')
            plt.title(c)
        if 'window' in c:
            plt.figure()
            sns.swarmplot('drug', 'nFeatures',\
                          data = consistent_features_grouped.get_group(c), hue = 'nworms')
            plt.title(c)
    
    #Calculate  euclidean distance   
    #fill nans and normalise the data
    FeatMatFinal_nona= FeatMatFinal.fillna(FeatMatFinal.mean(axis=0))
    FeatMatZ = pd.DataFrame(stats.zscore(FeatMatFinal_nona, ddof=1, axis=0), \
                            columns = FeatMatFinal_nona.columns,\
                            index = FeatMatFinal_nona.index)
    
    groupedFeatMatZ = FeatMatZ.groupby(['date', 'worm_number', 'drug', 'window']) 
    FeatMatZ_iterator = list(itertools.product(*metadata_dict.values()))
    EucDist= pd.DataFrame()
    for i in FeatMatZ_iterator:
        control_group = tuple(s if type(s)!=str else control for s in i)
        try:
            EucDist = EucDist.append(pd.Series({'eDist': euclidean_distances(np.array([groupedFeatMatZ.get_group(i).mean(axis=0).values,\
                                groupedFeatMatZ.get_group(control_group).mean(axis=0).values]))[0,1],\
                                                'metadata': i}),\
                                                ignore_index=True)
        except KeyError:
            print('no data for {} worms on {} drug on {}'.format(n, drug, date))
    
    EucDist[['date', 'worm_number', 'drug', 'window']] = pd.DataFrame(EucDist.metadata.tolist(), index=EucDist.index)
    
    #make a figure
    g = sns.FacetGrid(EucDist, col = 'window', hue = 'worm_number')
    g.map(sns.swarmplot, 'drug', 'eDist', order = metadata_dict['drug']).add_legend()
    plt.savefig(os.path.join(save_dir, 'euclideanDistance.png'))
    

    #compare 3 x 1worms vs 3 x 5worms
    N_totest = 3
    
    #iterate over groups and just take N_totest and combine onto datafrae
    slimFeatMat = pd.DataFrame()
    errorlog = []
    for i in FeatMatZ_iterator:
        try:
            slimFeatMat = slimFeatMat.append(FeatMatGrouped.get_group(i).sample(N_totest, random_state=1, replace=False))
        except Exception as error:
            print (error)
            errorlog.append((i, error))
    

    slimFeatMatGrouped = slimFeatMat.groupby(list(slimFeatMat.index.names))        
    #now calculate stats on slimFeatMat     
    pValsDF_slim, bhDF_slim = statsFeat(slimFeatMatGrouped, 'normal', 0.1, control)
    bhDF_slim= bhDF.reset_index()
    bhDF_slim['drug'] = [list(i) for i in bhDF_slim.metadata.str if i.dtype == object][0]
    bhDF_slim['date'] = bhDF_slim.metadata.str[0]
    bhDF_slim['worm_number'] = bhDF_slim.metadata.str[1]
    bhDF_slim['window'] = bhDF_slim.metadata.str[-1]
    bhDF_slim =bhDF_slim.drop(columns = 'metadata')
    
    #plot the total number of sig feats by worm number and drug (with window as hue) for each date
    for date in metadata_dict['date']:
        sns.catplot(x = 'worm_number', \
                    y = 'sumSig',\
                    data = bhDF_slim[bhDF_slim['date']==date], \
                    hue = 'window', \
                    col = 'drug',\
                    kind ='bar', \
                    palette = 'colorblind'
                    )
        plt.savefig(os.path.join(save_dir, 'EqualN_T_test_number_sig_feats5mins_{}.tif'.format(date)),
                    bbox_inches='tight',
                    pad_inches=0.03)
    #and for all data combined
    sns.catplot(x = 'worm_number', \
                y = 'sumSig',\
                data = bhDF, \
                hue = 'window', \
                col = 'drug',\
                kind ='bar', \
                palette = 'colorblind',
                ci= 'sd'
                ) 
    plt.savefig(os.path.join(save_dir,'EqualN_T_test_number_sigfigs5mins_combined.png'))    
    
    #TODO PCA of the 15min and 5 min data


    
