#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 17:43:13 2018

@author: ibarlow
"""

""" Combining Old and New Antipsychotics Datasets from saved csv files"""

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import os

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
featuresZ = featMatAll.select_dtypes(include=['float64']).drop(columns = 'concentration')
descriptor = featMatAll.select_dtypes(include = object)
descriptor = pd.concat([descriptor, featMatAll['concentration'], featMatAll['date']],axis=1)
featMatZ = pd.DataFrame(stats.zscore(featuresZ, axis=0), columns = featuresZ.columns)

drugs = descriptor['drug']
concs = descriptor ['concentration']
dates= descriptor['date']

allDrugs = np.unique(drugs)
allConcs = np.unique(concs)
allDates = np.unique(dates)

#zscore
def z_score(df):
    return (df-df.mean())/df.std(ddof=0)

featuresZ = featMatAll.select_dtypes(include=['float64']).drop(columns = 'concentration')
featMatZ = featuresZ.apply(z_score)
featMatZ = featMatZ.fillna(featMatZ.mean(axis=0))


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
savedir =  '/Volumes/behavgenom$/Ida/Data/Antipsychotics/Combined_analysis/Figures'
PC_custom.PC12_plots(PC_df, [],[], cmap1, savedir, 'tif', 'concentration')
PCmean, PCsem = PC_custom.PC_av(PC_df, [], 'concentration')

PCJ.PC_trajGraded(PCmean, PCsem,['PC_1','PC_2'], [], savedir, '.png', 'concentration', start_end = False,\
                  cum_var = cumvar, legend = 'off')
PC_custom.PC_traj(PCmean, PCsem,['PC_1', 'PC_2'], [], savedir, 'tif', cmap1,[], start_end=False)

#plot the colorgradients
mkc.plot_color_gradients(cm, cm.keys())
plt.savefig(os.path.join(savedir, 'gradientDrugColors.png'))


#map the cmap1 on the drugs for the clustermap
lut = dict(zip(allDrugs, cmap1))
row_colors = featMatZ2['drug'].map(lut)

#clustemap
cg=sns.clustermap(featMatZ2.select_dtypes(include = 'float').drop(columns = 'concentration'),\
                  metric  = 'euclidean', cmap = 'inferno', method = 'average',\
                  row_colors = row_colors, vmin = -4, vmax =4)
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
         (featMatZ2['drug'][cg.dendrogram_row.reordered_ind]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8)
#plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) 
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
#save fig
plt.savefig(os.path.join(savedir, 'AllCompoundsClustermap.tif'), \
            dpi =150, bbox_inches = 'tight', pad_inches = 1)
plt.show()

clusterDF = pd.DataFrame(featMatZ2['drug'][cg.dendrogram_row.reordered_ind])
clusterDF = pd.concat([clusterDF, featMatZ2['concentration'][cg.dendrogram_row.reordered_ind].to_frame()], axis=1)
clusterDF['drug_conc'] = list(zip(clusterDF.drug, clusterDF.concentration))
#clusterDF = clusterDF.reset_index(drop=True)

sns.set_style('white')
plt.figure(figsize = (30,10))
ax = plt.imshow(cg.data2d.iloc[200:220,:], vmin=-4, vmax =4, aspect='auto')
ax.axes.set_yticks(np.arange(0,20,1))
ax.axes.set_yticklabels(clusterDF.iloc[200:220,:]['drug_conc'])
plt.tight_layout()
plt.savefig(os.path.join(savedir, 'SelectRowsClustermap.tif'), dpi =200, bbox_inches = 'tight')

#and DMSO as a comparison
plt.figure(figsize=(30,10))
plt.imshow(featMatZ2[featMatZ2['drug']=='DMSO'].select_dtypes(include='float'),\
           aspect='auto', vmin =-4, vmax = 4)



#make a figure of the colorbar
colors = [(v) for k,v in lut.items()]
    #plot separately
plt.figure()
ax = plt.imshow([colors])
ax.axes.set_xticklabels(range(0,len(allDrugs),1))
ax.axes.set_xticklabels(lut.keys(), rotation = 90)
ax.axes.set_xticks(np.arange(0,len(allDrugs),1))
ax.axes.xaxis.set_ticks_position('top')
plt.savefig(os.path.join(savedir, 'drug_colors.png'),\
            bbox_inches='tight',dpi =150)
plt.close()

#look at the clustergram of just the new data
foo1 = featMatZ2['date']==20180906
bar1 =featMatZ2['date'] == 20181005
foo2 = featMatZ2['date'] ==20181011
bar2 = featMatZ2['date'] ==20181028

featMatZ3 = featMatZ2[foo1|bar1|foo2|bar2]

#remap rowcolors
row_colors= featMatZ3['drug'].map(lut)
cg=sns.clustermap(featMatZ3.select_dtypes(include = 'float').drop(columns = 'concentration'),\
                  metric  = 'euclidean', method = 'ward', cmap = 'inferno', \
                  row_colors = row_colors, vmin = -6, vmax =6)
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
         (featMatZ2['drug'][cg.dendrogram_row.reordered_ind]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8)
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) 
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
#save fig
plt.savefig(os.path.join(savedir, 'NewCompoundsClustermap.tif'), \
            dpi =150, bbox_inches = 'tight', pad_inches = 1)
plt.show()

#show example anticorrelating feature - this is acutally found later in the script
egFeat = 'relative_to_hips_radial_velocity_tail_tip_w_backward_90th'

plt.figure()
sns.set_style ('whitegrid')
sns.swarmplot(x='drug', y=egFeat, data=featMatZ3, hue = 'concentration',palette = 'Blues')
plt.xticks(rotation = 25)
plt.show()
plt.savefig(os.path.join(savedir, 'combinedNew' + egFeat + 'swarm.png'), bbox_inches = 'tight', dpi = 200)

#%% now add on all the statistics
    #this will be good to see if there are any specific features the new drugs affect

#do the stats separately for each replicate (date)
uniqueDates = list(np.unique(dates))
uniqueDrugs = list(np.unique(drugs))

#compile matrix of p-values for each rep
pVals = {}
for date in uniqueDates:
    pVals[date] = pd.DataFrame()
    featMatDate = featMatAll[featMatAll['date'] == date]
    control = featMatDate[featMatDate['drug'] == 'DMSO']
    for drug in uniqueDrugs:
        if drug == 'DMSO':
            continue
        else:
            featMatDrug = featMatDate[featMatDate['drug']==drug]
            testconcs = np.unique(featMatDrug['concentration'])
            for conc in testconcs:
                test=[]
                feats =[]
                for feat in featMatDrug.select_dtypes(include = ['float']).columns:
                    if feat == 'concentration':
                        continue
                    else:
                        test.append(stats.ttest_ind(featMatDrug[featMatDrug['concentration']==conc][feat], control[feat]))
                        feats.append(feat)
                ps = [(test[i][1]) for i in range(len(test))] #make into a list
                ps.append(drug)
                ps.append(conc)
                ps.append(date)
                
                feats.append('drug')
                feats.append('concentration')
                feats.append('date')
                
                temp = pd.DataFrame(ps).transpose()
                temp.columns = feats
                pVals[date] = pVals[date].append(temp)
                
                del conc, test, feat
            del testconcs, featMatDrug
    pVals[date].columns = feats
    pVals[date] = pVals[date].reset_index(drop=True)
    del featMatDate, control

#multiple comparison correction
import statsmodels.stats.multitest as smm

#now correct for multiple comparisons - bejamini hochberg procedure
bh_p={}
top_feats = {}
post_exclude = {}
sig_feats = {}
for rep in pVals:
    bh_p [rep] = pd.DataFrame()
    for cond in range(pVals[rep].shape[0]):
        reg, corrP, t,s =  smm.multipletests(pVals[rep].values[cond, :-3],\
                                             alpha=0.05, method = 'fdr_bh', \
                                             is_sorted = False, returnsorted = False)
        corrP = list(corrP)
        corrP.append(pVals[rep]['drug'].iloc[cond])
        corrP.append(pVals[rep]['concentration'].iloc[cond])
        corrP = pd.DataFrame(corrP).transpose()
        bh_p [rep]= bh_p[rep].append(corrP)
        del corrP
        #add in the feature names
    bh_p[rep].columns = feats[:-1]
    bh_p [rep]= bh_p[rep].reset_index(drop = True)

    #now can filter out the features that have no significant p-values
    top_feats[rep]= bh_p[rep].values[:,0:-2] <=0.05 #0.05 significance level
    top_feats[rep] = pd.DataFrame(data=top_feats[rep], columns = bh_p[rep].iloc[:,:-2].columns)
    top_feats[rep] = pd.concat([top_feats[rep], bh_p[rep].iloc[:,-2:]], axis=1 )
    
    post_exclude [rep]= [] #initialise
    sig_feats[rep] = []
    for feat in range(bh_p[rep].shape[1]-2):
        if np.sum(top_feats[rep].iloc[:,feat]) ==0:
            post_exclude[rep].append(bh_p[rep].columns[feat]) #all the features that show no difference
        else:
            sig_feats[rep].append((bh_p[rep].columns [feat], np.sum(top_feats[rep].iloc[:,feat])))
    
    #sort by most to least
    sig_feats[rep].sort(key =lambda tup:tup[1])
    sig_feats[rep].reverse()

#make a list of which features are significantly different for each drug
pVals2 = {}
for rep in sig_feats:
    pVals2[rep] = pd.DataFrame()
    for drug in uniqueDrugs:
        try:
            temp = top_feats[rep][top_feats[rep]['drug']==drug]
            temp2 = bh_p[rep][bh_p[rep]['drug'] == drug]
            conc = np.unique(temp['concentration'])
            for c in conc:
                temp3 = temp[temp['concentration']==c]
                temp4 = temp2[temp2['concentration'] ==c]
                #feats = list(temp3.columns[np.where(temp3)[1]])
                ps = temp4[temp3.columns[np.where(temp3)[1]]]
                ps['concentration']=  c
                ps['drug'] = drug
                pVals2[rep] = pVals2[rep].append(ps)
                del ps, temp3, temp4
            del temp, temp2, conc
        except TypeError:
            print(drug + ' not in this replicate')
            continue

#export the pVals2 as a csv
for rep in pVals2:
    pVals2[rep].to_csv(os.path.join(savedir, str(rep) + 'sigPs.csv' ))

#make some violin plots of the significant features
from feature_swarms import swarms

cmap2 =  sns.color_palette("Blues", len(allConcs))
sns.set_style('whitegrid')

for rep in pVals2:
    for feat in range(0,10):
        #swarms (str(rep), sig_feats[rep][feat][0], featMatZ2[featMatZ2['date']==rep], savedir, '.tif', cmap)
        swarms('all', sig_feats[rep][feat][0], featMatZ2 ,savedir, '.tif', cmap2)



# =============================================================================
# Things to do:
    # 1. Train classifier to distinguish between Atypical and typical drugs
    
    # 2. check DMSO controls on all dates and plot to compare
    
    # 3. Make a list of the drugs to repeat - DONE
# =============================================================================
#%% Compare features across reps
    
for date in allDates:
    plt.figure()
    plt.subplots(nrows =2, ncols=5, sharey = True)
    for feat in range(0,10):
        plt.subplot(2,5,feat+1)
        ax = sns.swarmplot(x = 'date', y = sig_feats[date][feat][0], \
                    data= featMatZ2[featMatZ2['drug']=='DMSO'], hue = 'date')
        plt.ylabel('')
        plt.ylim([-1 ,1])
        plt.title(sig_feats[date][feat][0])
        if feat>0:
            ax.legend_.remove()
        else:
            continue
plt.show()

#test all reps the same
test = 'd_relative_to_body_angular_velocity_neck_w_backward_90th'
for date in allDates:
    swarms(date, test, featMatZ2[featMatZ2['date']==date], savedir, '.tif', cmap1)

#do stats to compare PC2 for DMS0 control
DMSO_PC2 = pd.DataFrame()
for date in allDates:
    temp = PC_df[PC_df['drug']=='DMSO'][PC_df['date']==date]['PC_2'].to_frame()
    temp = temp.reset_index(drop=True).rename(columns = {'PC_2':date})
    
    DMSO_PC2 = pd.concat([DMSO_PC2, temp], axis=1)
    del temp


DMSOtest =[]
for date in allDates:
    DMSOtest.append(DMSO_PC2[date].values[~np.isnan(DMSO_PC2[date].values)])
        
p = stats.f_oneway(DMSOtest[0], DMSOtest[1], DMSOtest[2],DMSOtest[3], DMSOtest[4],\
                   DMSOtest[5] ,DMSOtest[6])

#there is a difference between the DMSO controls between the years
plt.figure()
sns.swarmplot(x='date', y='PC_2', data=PC_df[PC_df['drug']=='DMSO'], color = lut['DMSO'])
plt.text(1,0.3, '1way_anova, p=' + str(p[1]))
plt.savefig(os.path.join(savedir, 'PC2_1wayANOVA.png'))
plt.ylim([-0.5, 0.5])
plt.show()

#%% Implement contrastive PCA to  

from contrastive import CPCA
mdl = CPCA(n_components = 50)

#use No_Compound as background condition
foreground = np.array(featMatZ2[featMatZ2['drug']!='No_Compound'].select_dtypes(include='float').drop(columns = 'concentration'))
background = np.array(featMatZ2[featMatZ2['drug']=='No_Compound'].select_dtypes(include='float').drop(columns='concentration'))
Druglabels = featMatZ2[featMatZ2['drug']!='No_Compound']['drug'].to_frame().reset_index(drop=True)
Conclabels =featMatZ2[featMatZ2['drug']!='No_Compound']['concentration'].to_frame().reset_index(drop=True)
Datelabels = featMatZ2[featMatZ2['drug']!='No_Compound']['date'].to_frame().reset_index(drop=True)

#calculate CPCA with 50PCs
projected_data = mdl.fit_transform(foreground, background)

#and now plot to compare the alphas
cPC_df = {}
cPCmean={}
cPCsem = {}
for cpc in range(0,len(projected_data)):
    cPC_df[cpc] = pd.DataFrame(projected_data[cpc])
    cPC_df[cpc].columns = ['PC_' + str(i) for i in range (1,51)]
    cPC_df[cpc] = pd.concat([cPC_df[cpc], Druglabels, Conclabels, Datelabels], axis=1)

for cpc in range(0,len(projected_data)):
    PC_custom.PC12_plots(cPC_df[cpc], [],'alpha' + str(cpc), cmap1, savedir, 'tif', 'concentration')  
    cPCmean[cpc], cPCsem[cpc] = PC_custom.PC_av(cPC_df[cpc], [], 'concentration')

    PCJ.PC_trajGraded(cPCmean[cpc], cPCsem[cpc], ['PC_1','PC_2'], 'alpha' + str(cpc), savedir, \
                      '.png', 'concentration', start_end = False, cum_var = cumvar, legend= 'off')
    PC_custom.PC_traj(cPCmean[cpc], cPCsem[cpc],['PC_1', 'PC_2'],'alpha' + str(cpc), savedir, 'tif', cmap1,[], start_end=False)

#show all replicates for just a few drugs to demonstrate how consistent the behavioural responses are
scalex = 1/(cPC_df[3]['PC_1'].max() - cPC_df[3]['PC_1'].min())
scaley = 1/(cPC_df[3]['PC_2'].max() - cPC_df[3]['PC_2'].min())
xs = cPC_df[3]['PC_1']
ys = cPC_df[3]['PC_2']
cscale = np.arange(0, 5,1)

def specDrug12(selDrug,selConc, savedir, PC_DF):
    scalex = 1/(PC_DF['PC_1'].max() - PC_DF['PC_1'].min())
    scaley = 1/(PC_DF['PC_2'].max() - PC_DF['PC_2'].min())
    xs = PC_DF['PC_1']
    ys = PC_DF['PC_2']
    PC_DF ['PC_1'] = PC_DF['PC_1'].replace(PC_DF['PC_1'].values, xs*scalex)
    PC_DF ['PC_2'] = PC_DF['PC_2'].replace(PC_DF['PC_2'].values, ys*scaley)
    
    temp_1 = PC_DF[PC_DF['drug']!=selDrug[0]]
    plt.figure()
    plt.scatter(x = temp_1['PC_1'], y= temp_1['PC_2'], color = 'grey', alpha = 0.5, label = None)
    #plt.scatter(x=PC_DF[PC_DF['drug']=='DMSO']['PC_1'], y = PC_DF[PC_DF['drug']=='DMSO']['PC_2'], color = lut['DMSO'])
    
    for drug in selDrug:
        bar = PC_DF['drug'] ==drug
        foo = PC_DF['concentration'] == float(selConc)
        temp2 = PC_DF[bar&foo].reset_index(drop=True)
        
        plt.scatter(x= temp2['PC_1'], y = temp2['PC_2'], color = lut[drug], label=drug)#, 'c': cscale})#,\
    plt.xlim([-0.4, 0.4])
    plt.ylim([-0.4, 0.4])
    plt.axis('equal')

    plt.savefig(os.path.join(savedir, '_'.join(selDrugs) + str(selConc) + 'PC12.png'), dpi = 200)
    
selDrugs = ['Haloperidol', 'Sodium Valproate', 'Lamotrigine']
specDrug12(selDrugs,10, savedir, cPC_df[3])   

PCJ.PC_trajGraded(cPCmean[cpc], cPCsem[cpc], ['PC_5','PC_6'], 'alpha' + str(cpc), savedir,\
                  '.png', 'concentration', start_end = False, cum_var = cumvar, legend= 'off')


#%%only plot the old antipsychotics as trajectory
oldDrugs = ['DMSO', 'Chlopromazine hydrocholoride', 'Clozapine', 'Amisulpride', 'Aripiprazol',\
         'Olanzapine', 'Raclopride', 'Risperidone']
oldPC_DF = pd.DataFrame()
for drug in oldDrugs:
    temp = cPC_df[3][cPC_df[3]['drug']==drug]
    oldPC_DF = oldPC_DF.append(temp, sort=True)

cPCmeanOld, cPCsemOld = PC_custom.PC_av(oldPC_DF, [], 'concentration')

PCJ.PC_trajGraded(cPCmeanOld, cPCsemOld,['PC_1', 'PC_2'],  'old', savedir, \
          '.png', 'concentration', start_end = False, cum_var = cumvar, legend= 'off')

#%%
#and do stats
#do stats to compare PC2 for DMS0 control
cDMSO_PC2 = {}
cDMSOtest = {}
p = []
for cpc in cPC_df:
    cDMSO_PC2[cpc] = pd.DataFrame()
    for date in allDates:
        temp = cPC_df[cpc][cPC_df[cpc]['drug']=='DMSO'][cPC_df[cpc]['date']==date]['PC_2'].to_frame()
        temp = temp.reset_index(drop=True).rename(columns = {'PC_2':date})
        
        cDMSO_PC2[cpc] = pd.concat([cDMSO_PC2[cpc], temp], axis=1)
        del temp
    
    cDMSOtest [cpc]=[]
    for date in allDates:
        cDMSOtest[cpc].append(cDMSO_PC2[cpc][date].values[~np.isnan(cDMSO_PC2[cpc][date].values)])
        
        
    p.append(stats.f_oneway(cDMSOtest[cpc][0], cDMSOtest[cpc][1], cDMSOtest[cpc][2],\
                            cDMSOtest[cpc][3], cDMSOtest[cpc][4],cDMSOtest[cpc][5],\
                            cDMSOtest[cpc][6]))

#for all values of alpha the difference between DMSO at PC2 disappears
#to visually compare DMSO responses across dates
    plt.figure()
    sns.swarmplot(x='date', y='PC_2', data=cPC_df[cpc][cPC_df[cpc]['drug']=='DMSO'], color = lut['DMSO'])
    plt.ylim([-0.5, 0.5])
    plt.text(2, 0.5, '1way ANOVA, p = ' + str(p[cpc][1]))
    plt.savefig(os.path.join(savedir, 'cPC2_DMSOstats_' + 'alpha'+ str(cpc) + '.png'))
    plt.close('all')
    
#%% Now to try training a classifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#have to use the data transformed back into PC space to allow for differences between
    #years
    
#first try LDA to discriminate between typical and atypical antipsychotics and DMSO control
    #use the final alpha for cPC_df
cpc = 3

#add extra label to data frame of atypical vs typical antipsychotic
#use these features for classification


#make a conditions dataframe
conds = pd.concat([cPC_df[cpc].pop('drug'), cPC_df[cpc].pop('concentration'), cPC_df[cpc].pop('date')], axis =1)

#make separate dataframe classifying typical vs atypical vs DMSO control
conds2 = pd.DataFrame()
for line in conds.iterrows():
    line2 = line[1].to_frame().transpose()
    if line[1]['drug'] == 'Chlopromazine hydrocholoride' or line[1]['drug'] == 'Haloperidol':
        line2['class'] = 'typical'
    elif line[1]['drug'] == 'DMSO':
        line2 ['class'] ='control'
    else:
        line2['class'] = 'atypical'
    line2['drug2'] = line[1]['drug'] + str(line[1]['concentration'])
    conds2 = conds2.append(line2)
    del line2

#look at scatter plot of atypical, control and typical drugs

plt.figure()
sns.kdeplot(cPC_df2_2[cPC_df2_2['class']=='typical']['PC_1'], cPC_df2_2[cPC_df2_2['class']=='typical']['PC_2'],\
                 shade = True, shade_lowest = False, cmap = 'Reds', alpha = 0.7, label = 'typical')
sns.kdeplot(cPC_df2_2[cPC_df2_2['class']=='atypical']['PC_1'], cPC_df2_2[cPC_df2_2['class']=='atypical']['PC_2'],\
                 shade = True, shade_lowest = False, cmap = 'Blues', alpha = 0.7, label = 'atypical')
sns.kdeplot(cPC_df2_2[cPC_df2_2['class']=='control']['PC_1'], cPC_df2_2[cPC_df2_2['class']=='control']['PC_2'],\
                 shade = True, shade_lowest = False, cmap = 'Greys', alpha = 0.7, label = 'control')
plt.axis('equal')
plt.xlim([-50,50])
plt.ylim([-50, 50])
plt.show()
plt.savefig(os.path.join(savedir, 'DensityPCdrugClass.png'), dpi=200)


#remove data from 12042017 as the controls for this dataset are off
cPC_df2_2 = cPC_df2[cPC_df2['date']!=12042017]

#make into arrays for classification

#for splitting the data and crossvalidation
    #shuffle 10-fold cross validation
sss = StratifiedShuffleSplit(n_splits = 10, train_size=0.8)

#set liniar discriminant classifier
clf = LinearDiscriminantAnalysis(n_components = None, priors = None, \
                                 shrinkage = 'auto', solver = 'eigen')

#initialise dictionary of lists for storing the multiple rounds of cross validation
#and different number of principal components
score = {}
class_scaling = {}
class_means = {}
coefs = {}
y = np.array(cPC_df2_2['class'])
for i in range(1,51):
    X = np.array(cPC_df2_2.iloc[:,:i].select_dtypes(include= 'float'))
    
    score[i] = []
    class_scaling[i] = []
    class_means[i] = []
    coefs[i]=[]
    #start cross-validation
    for train_index, test_index in sss.split(X, y):
        #split the testing and training data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
       
        #then do LDA using these training/testing sets
            #store the transformed scaling for each set
        clf.fit(X_train, y_train)
        
        #store the classification score
        score[i].append(clf.score(X_test, y_test))
       
        #weightings for each feature for each run
        coefs[i].append(clf.coef_[0])
        
        #scaling matrix - from PCA of feature space
        class_scaling[i].append(clf.scalings_)
        
        #mean (raw) for each class
        class_means[i].append(clf.means_)
    
        del X_train, X_test, y_train, y_test
    del X
        
    classes = clf.classes_

#make a plot of the score vs number of PCs
scoreAv =[]
scoreStd = []
for i in score:
    scoreAv.append(np.mean(score[i]))
    scoreStd.append(np.std(score[i]))

from sklearn.utils import shuffle
yshuffle = shuffle(y)

scoreShuffle = {}
class_scalingShuffle = {}
class_meansShuffle = {}
coefsShuffle = {}
for i in range(1,51):
    X = np.array(cPC_df2_2.iloc[:,:i].select_dtypes(include= 'float'))
    
    scoreShuffle[i] = []
    class_scalingShuffle[i] = []
    class_meansShuffle[i] = []
    coefsShuffle[i]=[]
    #start cross-validation
    for train_index, test_index in sss.split(X, yshuffle):
        #split the testing and training data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = yshuffle[train_index], yshuffle[test_index]
       
        #then do LDA using these training/testing sets
            #store the transformed scaling for each set
        clf.fit(X_train, y_train)
        
        #store the classification score
        scoreShuffle[i].append(clf.score(X_test, y_test))
       
        #weightings for each feature for each run
        coefsShuffle[i].append(clf.coef_[0])
        
        #scaling matrix - from PCA of feature space
        class_scalingShuffle[i].append(clf.scalings_)
        
        #mean (raw) for each class
        class_meansShuffle[i].append(clf.means_)
    
        del X_train, X_test, y_train, y_test
    del X
        
    classes = clf.classes_

scoreAvShuffle =[]
scoreStdShuffle = []
for i in score:
    scoreAvShuffle.append(np.mean(scoreShuffle[i]))
    scoreStdShuffle.append(np.std(scoreShuffle[i]))

plt.figure()
plt.errorbar(range(1,51), scoreAv, yerr= scoreStd)
plt.errorbar(range(1,51), scoreAvShuffle, yerr = scoreStdShuffle)
plt.show()

#Hmm same score for shuffled data... (!)

#try an SVM instead


#%% tSNE embedding


