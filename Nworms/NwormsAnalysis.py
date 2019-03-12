#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:37:29 2018

@author: ibarlow
"""

""" Analysis script to testing N worm effects"""

%cd /Users/ibarlow/Documents/GitHub/pythonScripts/Functions

import pandas as pd
import os
import numpy as np
import displayfiles as gfn

#find the replicates in the folder
foldIn = '/Volumes/behavgenom$/Ida/Data/NwormTests/'
reps = [x for x in os.listdir(foldIn) if x.startswith('201')]

#and the data
repData = {}
metadata = {}
for item in reps:
    if item.endswith('.tsv'):
        continue
    else:
        repData[item]= gfn.displayfiles('featuresN.hdf5',inputfolder = os.path.join(foldIn, item),\
              outputfile = '')
        #load metadata
        metadata[item]= pd.read_csv(os.path.join(foldIn, item, 'AuxiliaryFiles', 'metadata.csv'))
        metadata[item] = metadata[item].dropna(axis=0, how='all')
        metadata[item]['fname'] = repData[item][1]
        
#match up metadata with conditions - number of worms and drugs
#and load the features_stats which are the plate summaries
featMat= {}
for rep in repData:
    featMat[rep] = pd.DataFrame()
    for line in range(0,len(repData[rep][1])):
        try:
            with pd.HDFStore(repData[rep][1][line], 'r') as fid:
                temp = fid['/features_stats']['value'].to_frame().transpose()
                temp.columns = fid['/features_stats']['name']
                
                if repData[rep][1][line] == metadata[rep]['fname'][line]:
                    #print (repData[rep][1][line] + ' file match')
                    temp['drug']= metadata[rep][' drug type'][line]
                    temp['concentration'] = metadata[rep][' drug concentration'][line]
                    temp['Nworms'] = metadata[rep][' worm number'][line]
                    temp['setN'] = metadata[rep][' set number'][line]
                    temp['date'] = str(rep)
                    
                else:
                    print (repData[rep][1][line] + ' filename mismatch at file ' + line)
                    continue
                
                featMat[rep] = featMat[rep].append(temp, sort= True)
                
                del temp
        except KeyError:
            print (repData[rep][1][line] + ' has no features data')

    featMat[rep] = featMat[rep].reset_index(drop=True)
    
    #drop all nan columns
    featMat[rep] = featMat[rep].dropna(axis=1, how='all')
    
    #fill the nan values of features with nans to avoid z scoring errors
    featMat[rep]= featMat[rep].fillna(featMat[rep].mean(), axis = 0)
  
    
#%% Now to filter, normalise and look at the results for 1 vs 5 vs 10 worms    
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

#zscore
def z_score(df):
    return (df-df.mean())/df.std(ddof=0)

#combined datasets before Zscoring
featMatAll = pd.DataFrame()
for rep in featMat:
    featMatAll = featMatAll.append(featMat[rep])
#drop haloperidol data
featMatAll = featMatAll[featMatAll['drug'] != 'Haloperidol']
featMatAll = featMatAll.reset_index(drop=True)

#zscore and make lists of conditions on combined data
conds = featMatAll[['drug', 'concentration', 'Nworms', 'setN', 'date']]   
featMatZ = pd.DataFrame(z_score(featMatAll.select_dtypes(include = 'float32')),\
            columns = featMatAll.select_dtypes(include='float32').columns)
featMatZ2 = pd.concat([featMatZ, conds], axis=1)

#drop columns with too many nans
featMatZ2 = featMatZ2.drop(columns = featMatZ2.columns[featMatZ2.isna().sum()>featMatZ2.shape[0]/2-1])

#%%
#now can do PCA
X = np.array(featMatZ)

pca = PCA()
X2 = pca.fit_transform(X)

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
PC_df['drug'] = conds['drug']
PC_df['concentration'] = conds['concentration']
PC_df['Nworms'] = conds['Nworms']
PC_df['date'] = conds['date']
#PC_df['date'] = date_all

#components that explain the variance
    #make a dataframe ranking the features for each PC and also include the explained variance (z-normalised)
        #in separate dataframe called PC_sum
PC_feat = [] #features
PC_sum =[] #explained variance
for PC in range(0, len(PCname)):
    PC_feat.append(list(featMatZ.columns[np.argsort(pca.components_[PC]**2)]))
    PC_sum.append(list((pca.components_[PC])/ np.sum(abs(pca.components_[PC]**2))))
    
#dataframe containing standards scores of contibution of each feature
        #rows correspond to PCs
PC_vals = pd.DataFrame(data= PC_sum, columns = featMatZ.columns)

#okay so now can plot as biplot
plt.figure()
for i in range(0,1):
    plt.arrow(0,0, PC_vals.iloc[0,:][PC_feat[0][-1-i]]*100, \
              PC_vals.iloc[1,:][PC_feat[0][-1-i]]*100,color= 'b')
    plt.arrow(0,0, PC_vals.iloc[0,:][PC_feat[1][-1-i]]*100,\
              PC_vals.iloc[1,:][PC_feat[1][-1-i]]*100, color='r')
    plt.text(PC_vals.iloc[0,:][PC_feat[0][-1-i]] + 7,\
             PC_vals.iloc[1,:][PC_feat[0][-1-i]] - 3, PC_feat[0][-1-i],\
             ha='center', va='center')
    plt.text(PC_vals.iloc[0,:][PC_feat[1][-1-i]]+5, PC_vals.iloc[1,:][PC_feat[1][-1-i]]+3,\
         PC_feat[1][-1-i], ha='center', va='center')
plt.xlim (-10, 10)
plt.ylim (-10, 10)
plt.xlabel('%' + 'PC_1 (%.2f)' % (pca.explained_variance_ratio_[0]*100), fontsize = 16)
plt.ylabel('%' + 'PC_2 (%.2f)' % (pca.explained_variance_ratio_[1]*100), fontsize = 16)
plt.show()
plt.savefig(os.path.join(foldIn, 'Figures', 'agar_biplot.png'))

import PCA_analysis as PC_custom 
sns.set()
cmap1 = sns.color_palette('tab10',len(np.unique(conds['drug'])))
allDrugs = np.unique(conds['drug'])

#make the PC plots
savedir =  os.path.join(foldIn, 'Figures')
for rep in featMat:   
    PC_custom.PC12_plots(PC_df.loc[PC_df['date']==rep], 1 ,rep + '_1worm', \
                         cmap1, savedir, 'tif', 'Nworms', False)
plt.close('all')

#combined
PC_custom.PC12_plots(PC_df, 5,'combined_5worms', cmap1, savedir, 'tif', 'Nworms', False)

#all the data
PC_custom.PC12_plots(PC_df, [],'alldata', \
                         cmap1, savedir, 'tif', [], False)

#make another version of dataframe with the drug column also containing the Nworms
PC_df['drug2'] = list(zip(PC_df.drug, PC_df.Nworms))
conds['drug2'] = PC_df['drug2']
PC_df = PC_df.drop(columns = 'drug')
PC_df = PC_df.rename(columns ={ 'drug2': 'drug'})

cmap2 = sns.color_palette('tab20',len(np.unique(PC_df['drug'])))

PC_custom.PC12_plots(PC_df, [], 'alldata_allworms', cmap2, savedir,'tif', [], False)

PCmean, PCsem = PC_custom.PC_av(PC_df, [], 'Nworms')

PC_custom.PC_traj(PCmean, PCsem,['PC_1', 'PC_2'], [], savedir, 'tif', cmap1,[], start_end=False)

#make graded colormap
from matplotlib.colors import LinearSegmentedColormap

cmapGraded = []
for item in cmap1:
    cmapGraded.append([(1,1,1), (item)]) #,(0.3,0.3,0.3)])
    
lutGraded = dict(zip(np.unique(conds['drug']), cmapGraded))
cm={}
for drug in lutGraded:
    cmap_name = drug
    # Create the colormap
    cm[drug] = LinearSegmentedColormap.from_list(
        cmap_name, lutGraded[drug], N=20)
    plt.register_cmap(cmap = cm[drug])
    #plt.register_cmap(name=drug, data=LinearSegmentedColormap.from_list())  # optional lut kwarg

#have a look at the colors
import make_colormaps as mkc
mkc.plot_color_gradients(cmap_list=cm, drug_names = lutGraded.keys())

#make the figure
plt.close()
plt.figure()
cscale = np.arange(1,len(np.unique(PC_df['Nworms']))+1,1)
xscale = 1/(PCmean.max()['PC_1']-PCmean.min()['PC_1'])
yscale = 1/(PCmean.max()['PC_2']-PCmean.min()['PC_2'])
for drug in allDrugs:
    if drug == 'Haloperidol':
        continue
    else:
        temp = PCmean[PCmean['drug']==drug]
        plt.scatter(temp['PC_1']*xscale, temp['PC_2']*yscale, \
                    cmap = plt.get_cmap(drug), c=cscale, label = drug, vmin = 0, zorder =1)
        del temp
plt.pause(0.1)
plt.axis('scaled')
plt.xlim(-1,1)
plt.ylim(-1, 1)
plt.legend()
for drug in allDrugs:
    if drug == 'Haloperidol':
        continue
    else:
        plt.errorbar(x = PCmean[PCmean['drug']==drug]['PC_1']*xscale, \
                     y =PCmean[PCmean['drug']==drug]['PC_2']*yscale,\
                     xerr = PCsem[PCsem['drug']==drug]['PC_1']*xscale,\
                     yerr=PCsem[PCsem['drug']==drug]['PC_2']*yscale,\
                     color = [0.9, 0.9, 0.9], label = None, zorder = -1)
plt.savefig(os.path.join(savedir, 'N_wormsPCA.png'), dpi= 100)

#The largest variance may actual be the day of the experiment and so use contrastive PCA
#to remove the background variance

#%% do stats to compare the p values between 1 and 5 worms

from scipy.spatial import distance
drugs_to_test = list(np.unique(PC_df['drug']))
N_totest = [1, 5, 10]
controlGroup = featMatAll.groupby(by='drug').get_group('DMSO')

pVals = pd.DataFrame()
feats = list(featMatAll.select_dtypes(include = 'float32').columns)
for drug in drugs_to_test:
    drugTemp = featMatAll.groupby(by='drug').get_group(drug)
    for n in N_totest:
        controls = controlGroup.groupby(by='Nworms').get_group(n)
        test = drugTemp.groupby(by='Nworms').get_group(n)
        results = []
        for feature in feats:
            results.append(stats.ttest_ind(controls[feature], test[feature]))
    
        ps = [float(results[i][1]) for i in range(len(results))]
        ps.append(drug)
        ps.append(n)
        
        pVals = pVals.append(pd.DataFrame(ps).transpose())
        
        del controls, test, results, ps
#add on the feature names
feats.append('drug')
feats.append('Nworms')     
pVals.columns = feats
pVals = pVals.reset_index(drop=True)  

#correct for multiple comparisons
import statsmodels.stats.multitest as smm
bh_p = pd.DataFrame()
for row in range(pVals.shape[0]):
    reg,corrP,t,s = smm.multipletests(pVals.drop(columns = ['drug', 'Nworms']).values[row,:], \
                                      alpha= 0.05, method= 'fdr_bh', is_sorted=False,\
                                      returnsorted=False)
    corrP = list(corrP)
    corrP.append(pVals['drug'].iloc[row])
    corrP.append(pVals['Nworms'].iloc[row])
    
    bh_p = bh_p.append(pd.DataFrame(corrP).transpose())
    
    del corrP, reg, t, s

bh_p = bh_p.reset_index(drop=True)
bh_p.columns = feats

sig_feats = pd.concat([pd.DataFrame(data = bh_p.drop(columns = ['Nworms', 'drug']).values<0.05, columns = feats[:-2]),\
                       bh_p.iloc[:,-2:]], axis=1)

#how many features are significantly differerent?
print (pd.concat([sig_feats[['drug', 'Nworms']], sig_feats.sum(axis=1)], axis=1, ignore_index=True))

from feature_swarms import swarms

#find example features which are significant for 1 worm but not for 5 worms
Plt_Egs = sig_feats.groupby('drug')
diff_feats = {}
for drug in drugs_to_test:
     
    temp = Plt_Egs.get_group(drug)
    temp = temp[temp['Nworms']!=10].select_dtypes(include='bool')
    #only keep sig columns that are different
    diff_feats = temp.columns[temp.select_dtypes(include=['bool']).sum(axis=0)==1]
    shared_feats = temp.columns[temp.select_dtypes(include=['bool']).sum(axis=0)==2]
    diff_feats[drug] =

for i in range(0,10):
    plt.figure()
    sns.swarmplot(x='Nworms', y=shared_feats[i], data=featMatZ2.groupby('drug').get_group(drug), \
                          palette = cmap1)
    plt.show()


#%% 

from contrastive import CPCA
import PC_traj as PCJ
mdl = CPCA(n_components = 50)

#use No_Compound as background condition
foreground = np.array(featMatZ2[featMatZ2['drug']!='DMSO'].select_dtypes(include='float32'))
background = np.array(featMatZ2[featMatZ2['drug']=='DMSO'].select_dtypes(include='float32'))
Druglabels = featMatZ2[featMatZ2['drug']!='DMSO']['drug'].to_frame().reset_index(drop=True)
Conclabels =featMatZ2[featMatZ2['drug']!='DMSO']['concentration'].to_frame().reset_index(drop=True)
Datelabels = featMatZ2[featMatZ2['drug']!='DMSO']['date'].to_frame().reset_index(drop=True)
Nwormlabels = featMatZ2[featMatZ2['drug']!='DMSO']['Nworms'].to_frame().reset_index(drop=True)

#calculate CPCA with 50PCs
projected_data = mdl.fit_transform(foreground, background)

#and now plot to compare the alphas
cPC_df = {}
cPCmean={}
cPCsem = {}
for cpc in range(0,len(projected_data)):
    cPC_df[cpc] = pd.DataFrame(projected_data[cpc])
    cPC_df[cpc].columns = ['PC_' + str(i) for i in range (1,51)]
    cPC_df[cpc] = pd.concat([cPC_df[cpc], Druglabels, Conclabels, Datelabels, Nwormlabels], axis=1)

for cpc in range(0,len(projected_data)):
    PC_custom.PC12_plots(cPC_df[cpc], 5,'_5worm' + str(cpc), cmap1, savedir, 'tif', 'Nworms', False)  
    cPCmean[cpc], cPCsem[cpc] = PC_custom.PC_av(cPC_df[cpc], [], 'Nworms')

    PCJ.PC_trajGraded(cPCmean[cpc], cPCsem[cpc], ['PC_1','PC_2'], 'alpha' + str(cpc), savedir, \
                      '.png', 'Nworms', start_end = False, cum_var = cumvar, legend= 'off')
    PC_custom.PC_traj(cPCmean[cpc], cPCsem[cpc],['PC_1', 'PC_2'],'alpha' + str(cpc), savedir, 'tif', cmap1,[], start_end=False)



#%% now calculate euclidean distance between drugs at different worm numbers

edist = pd.DataFrame(distance.squareform(distance.pdist(featMatZ.values)))
edist = pd.concat([edist, conds],axis=1)



#%% Hierachical clustering


allDrugs = np.unique(condsAll['drug'])
lut = dict(zip(allDrugs, cmap1))
rowColors = condsAll['drug'].map(lut)
cg = sns.clustermap(featMatZall, row_colors=  rowColors)

# =============================================================================
# Things to do:
# 1. Hierachical clustering and clustermap to show that for all worm numbers a similar
# or not effect is seen

# 2. Euclidean distance between drugs and DMSO for each worm number

# 3. The standard deviation and variance comparing worm number

# 4.
# =============================================================================
