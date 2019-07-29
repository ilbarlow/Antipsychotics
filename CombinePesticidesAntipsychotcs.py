#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:56:14 2018

@author: ibarlow

Combining pesticides and antipsychotics
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

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

#make a descriptor df
descriptor = featMatAll.select_dtypes(include = object)
descriptor = pd.concat([descriptor,featMatAll['concentration'], featMatAll['date']],axis=1)

drugs = descriptor['drug']
concs = descriptor ['concentration']
dates= descriptor['date']
MoAs = descriptor['MoAGeneral']

allDrugs = np.unique(drugs)
allConcs = np.unique(concs)
allDates = np.unique(dates)
allMoAs = np.unique(MoAs[MoAs.isna()==False])

#remove features with too many nans
featMatAll = featMatAll.drop(columns=\
                             list(featMatAll.columns[featMatAll.isna().sum()>featMatAll.shape[0]/10]))

#zscore
def z_score(df):
    return (df-df.mean())/df.std(ddof=0)

featuresZ = featMatAll.select_dtypes(include=['float64']).drop(columns = 'concentration')
featMatZ = featuresZ.apply(z_score)
featMatZ = featMatZ.fillna(featMatZ.mean(axis=0))


featMatZ2 = pd.concat([featMatZ, drugs, concs, dates, MoAs], axis=1)


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
    #make a dataframe ranking the features for each PC and also include the explained variance
        #in separate dataframe called PC_sum
PC_feat = [] #features
PC_sum =[] #explained variance
for PC in range(0, len(PCname)):
    sortPCs = np.flip(np.argsort(pca.components_[PC]**2), axis=0)
    PC_feat.append(list(featuresZ.columns[sortPCs]))
    weights = (pca.components_[PC]**2)/np.sum(pca.components_[PC]**2)
    PC_sum.append(list(weights))
    
#dataframe containing standards scores of contibution of each feature
        #rows correspond to PCs
PC_vals = pd.DataFrame(data= PC_sum, columns = featuresZ.columns) #march 2019 this is wrong!!!

#okay so now can plot as biplot
plt.figure()
for i in range(98,99):
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
savedir = '/Volumes/behavgenom$/Ida/Data/Antipsychotics/Combined_analysis/PestFigures'

#reset figure settings
sns.set_style('white')

#graded colormap
cmapGraded = mkc.get_colormaps(RGBsteps =9, thirdColorSteps = 3, n_bins=20)

lutGraded = dict(zip(allDrugs, cmapGraded))
for drug in lutGraded:
    plt.register_cmap(name = drug, cmap = lutGraded[drug])

mkc.plot_color_gradients(lutGraded.values(), lutGraded.keys())
plt.savefig(os.path.join(savedir, 'DrugColorsGraded.png'), dpi=200)

#single color map
lut = {}
for item in lutGraded:
    extract = np.round(sns.color_palette(item,1),2)
    lut[item] = [float(i) for i in extract[0]]

#map of drug colors for clustermap
row_colors = featMatZ2['drug'].map(lut)
#clustemap
cg=sns.clustermap(featMatZ,\
                  metric  = 'euclidean', cmap = 'inferno', method = 'average',\
                  row_colors = row_colors, vmin = -6, vmax =6)
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

#make a figure of colors for a legend
#make a figure of the colorbar
colors = [(v) for k,v in lut.items()]
    #plot separately
plt.figure(figsize = (30,10))
ax = plt.imshow([colors])
ax.axes.set_xticklabels(range(0,len(allDrugs),1))
ax.axes.set_xticklabels(lut.keys(), rotation = 90)
ax.axes.set_xticks(np.arange(0,len(allDrugs),1))
ax.axes.xaxis.set_ticks_position('top')
plt.savefig(os.path.join(savedir, 'drug_colors.png'),\
            bbox_inches='tight',dpi =150)
plt.close()

PC_custom.PC12_plots(PC_df, [],[], lut, savedir, 'tif', 'concentration')
PCmean, PCsem = PC_custom.PC_av(PC_df, [], 'concentration')

PCJ.PC_trajGraded(PCmean, PCsem, ['PC_1', 'PC_2'], [], savedir, '.png', 'concentration', start_end = False,\
                  cum_var = cumvar, legend = 'off')


# =============================================================================
# To do:
# 1. Do contrastive PCA 
# 2. Label antipsychotics (typical, atypical, and test compounds) and pesticides
    # and look at the distribution of these compounds across multiple principal components
# 3. Is it possible to train a classifier to differentiate between antipsychotics and pesticides?
    
# 4. tSNE embedding
    
    

# =============================================================================

#%% cPCA - could use df.groupby function here
from contrastive import CPCA

mdl = CPCA(n_components = 2)
foreground = np.array(featMatZ2[featMatZ2['drug']!='No_Compound'].select_dtypes(include='float').drop(columns = 'concentration'))
background = np.array(featMatZ2[featMatZ2['drug']=='No_Compound'].select_dtypes(include='float').drop(columns='concentration'))
Druglabels = featMatZ2[featMatZ2['drug']!='No_Compound']['drug'].to_frame().reset_index(drop=True)
Conclabels =featMatZ2[featMatZ2['drug']!='No_Compound']['concentration'].to_frame().reset_index(drop=True)
Datelabels = featMatZ2[featMatZ2['drug']!='No_Compound']['date'].to_frame().reset_index(drop=True)
MoAlabels = featMatZ2[featMatZ2['drug']!='No_Compound']['MoAGeneral'].to_frame().reset_index(drop=True)

#test and see what alpha looks best
mdl.fit_transform(foreground, background, plot=True, active_labels=Druglabels)
alpha1 = 1.34

#calculate CPCA with 50PCs
mdl = CPCA(n_components = 50)
projected_data = mdl.fit_transform(foreground, background)


#and now compare the alphas
cPC_df = {}
cPCmean={}
cPCsem = {}
for cpc in range(0,len(projected_data)):
    cPC_df[cpc] = pd.DataFrame(projected_data[cpc])
    cPC_df[cpc].columns = ['PC_' + str(i) for i in range (1,51)]
    cPC_df[cpc] = pd.concat([cPC_df[cpc], Druglabels, Conclabels, Datelabels, MoAlabels], axis=1)

for cpc in range(0,len(projected_data)):
    PC_custom.PC12_plots(cPC_df[cpc], [],'alpha' + str(cpc), lut, savedir, 'tif', 'concentration')  
    cPCmean[cpc], cPCsem[cpc] = PC_custom.PC_av(cPC_df[cpc], [], 'concentration')

    PCJ.PC_trajGraded(cPCmean[cpc], cPCsem[cpc], ['PC_1', 'PC_2'], 'alpha' + str(cpc), savedir, \
                      '.png', 'concentration', start_end = False, cum_var = cumvar, legend= 'off')
    plt.close('all')

#compare DMSO controls across experiments
#do stats to compare PC2 for DMS0 control
DMSO_PC2 = pd.DataFrame()
for date in allDates:
    temp = cPC_df[1][cPC_df[1]['drug']=='DMSO'][cPC_df[1]['date']==date]['PC_2'].to_frame()
    temp = temp.reset_index(drop=True).rename(columns = {'PC_2':date})
    
    DMSO_PC2 = pd.concat([DMSO_PC2, temp], axis=1)
    del temp

DMSOtest =[]
for date in allDates:
    DMSOtest.append(DMSO_PC2[date].values[~np.isnan(DMSO_PC2[date].values)])
 

p = stats.f_oneway(DMSOtest[0], DMSOtest[1], DMSOtest[2],DMSOtest[3], DMSOtest[4],\
                   DMSOtest[5], DMSOtest[6], DMSOtest[7], DMSOtest[8], DMSOtest[9],\
                   DMSOtest[10], DMSOtest[11], DMSOtest[12], DMSOtest[13], DMSOtest[14],\
                   DMSOtest[15])

#there is a difference between the DMSO controls between the years
plt.figure()
sns.swarmplot(x='date', y='PC_2', data=cPC_df[1][cPC_df[1]['drug']=='DMSO'], color = lut['DMSO'])
plt.text(1,0.1, '1way_anova, p=' + str(p[1]))
plt.savefig(os.path.join(savedir, 'PC2_1wayANOVA.png'))
plt.ylim([-0.1, 0.1])
plt.show()

#only plot the old antipsychotics as trajectory
oldDrugs = ['DMSO', 'Chlopromazine hydrocholoride', 'Clozapine', 'Amisulpride', 'Aripiprazol',\
         'Olanzapine', 'Raclopride', 'Risperidone']

#only plot pesticides
pestDrugs = list(np.unique(featmats['Pesticides']['drug']))
pestDrugs = pestDrugs[:-3] #drop DMSO and No_compound

#only plot new drugs
newDrugs = ['DMSO', 'Clozapine', 'Xanomeline', 'TAAR1', 'Sodium Valproate', 'Riluzole',\
            'Lamotrigine', 'Haloperidol']


#looks like there may be overlap between pesticides and antipsychotics so just plot these selected drugs

def specDrug12(selDrug, savedir, figname, PC_DFmean, PC_DFsem, scaling,lut, PCsToPlot, legOnOff):
    """function to only plot specific drugs as trajectory on top of all other drugs
    Input:
        selDrug: list of selected drugs
        savedir: where to save the figure
        PC_DFmean: the PCMean dataframe
        PC_DFsem:
        scaling: how to scale the data eg concentration
        lut: look up table of drug colors
        PCsToPlot = Principal components to plot
        
    Output:
        figure"""
    
    PC = [PCsToPlot[0]]
    PC.append(PCsToPlot[1])
    
    scalex = 1/(PC_DFmean[PC[0]].max() - PC_DFmean[PC[0]].min())
    scaley = 1/(PC_DFmean[PC[1]].max() - PC_DFmean[PC[1]].min())
    
    temp_1 = PC_DFmean[PC_DFmean['drug']!=selDrug[0]]
    plt.figure()
    plt.scatter(x = temp_1[PC[0]]*scalex, y= temp_1[PC[1]]*scaley, \
                color = [0.9, 0.9, 0.9], alpha = 0.5, label = None, zorder = -0.5)
    
    for drug in selDrug:
        cscale = np.arange(1, np.unique(PC_DFmean[PC_DFmean['drug']==drug][scaling]).shape[0]+1,1)
        bar = PC_DFmean['drug'] ==drug
        temp2 = PC_DFmean[bar].reset_index(drop=True)
        temp3 = PC_DFsem[bar].reset_index(drop=True)
        
        plt.errorbar(x=temp2[PC[0]]*scalex, y=temp2[PC[1]]*scaley, xerr= temp3[PC[0]]*scalex, yerr=temp3[PC[1]]*scaley,\
                     color = [0.7, 0.7, 0.7], zorder = 0, elinewidth=2, label=None)
        plt.scatter(x= temp2[PC[0]]*scalex, y = temp2[PC[1]]*scaley, \
                    color = lut[drug], label=drug, zorder =1)#, 'c': cscale})#,\
    plt.axis('scaled')
    plt.xlabel(PC[0])
    plt.ylabel(PC[1])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title(figname)
    if legOnOff.lower() == 'on':
        plt.legend(loc = 'best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, figname + '_' + str(PCsToPlot) + 'errobar.png'), dpi = 200)
    
#plot PC1 and PC2
for MoA in allMoAs:
    drugs = np.unique(cPC_df[1][cPC_df[1]['MoAGeneral']==MoA]['drug'])
    specDrug12(drugs, savedir, MoA, cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_1', 'PC_2'], 'On')

specDrug12(pestDrugs, savedir, 'pestDrugs', cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_1', 'PC_2'], 'Off')
specDrug12(oldDrugs, savedir, 'oldAntipsychotics', cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_1', 'PC_2'], 'On')
specDrug12(newDrugs, savedir, 'newAntipsychotics', cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_1', 'PC_2'], 'On')
plt.close('all')

#and PC3 and 4
for MoA in allMoAs:
    drugs = np.unique(cPC_df[1][cPC_df[1]['MoAGeneral']==MoA]['drug'])
    specDrug12(drugs, savedir, MoA, cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_3', 'PC_4'],'On')

specDrug12(pestDrugs, savedir, 'pestDrugs', cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_3', 'PC_4'], 'Off')
specDrug12(oldDrugs, savedir, 'oldAntipsychotics', cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_3', 'PC_4'], 'On')
specDrug12(newDrugs, savedir, 'newAntipsychotics', cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_3', 'PC_4'], 'on')
plt.close('all')

#all antipsychotics
allAntis = np.unique([oldDrugs, newDrugs])
specDrug12(allAntis, savedir, 'AllAntipsychotics', cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_1', 'PC_2'], 'Off')
specDrug12(allAntis, savedir, 'AllAntipsychotics', cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_3', 'PC_4'], 'Off')
specDrug12(allAntis, savedir, 'AllAntipsychotics', cPCmean[1], cPCsem[1], 'concentration', lut, ['PC_4', 'PC_6'], 'Off')

#%% figure out the variance explained - using the Rayleigh quotient

mdl = CPCA(n_components = 50)
projected_data = mdl.fit_transform(foreground, background,\
                                   alpha_selection = 'manual',  alpha_value=alpha1)
c_cov = mdl.bg_cov - alpha1* mdl.fg_cov

eigvals, eigvecs = np.linalg.eig(c_cov)

posEigs = np.argwhere(eigvals>0)
posEigvals= eigvals[posEigs]
posEigvecs = eigvecs[posEigs]

eig_pairs = [(np.abs(posEigvals[i]), posEigvecs[i]) for i in range(len(posEigvecs))]

#then sort - high to low
eig_pairs.sort(key =lambda tup:tup[0])
eig_pairs.reverse()

#rayleigh quotient - take top 3 eigenvectors
R=np.empty(4)
for i in range(0,4):
    t= np.dot(c_cov, eig_pairs[i][1].T)
    tt = np.dot(eig_pairs[i][1], t)
    r = sum(tt) / np.sum(np.square(eig_pairs[i][1]),axis=1)
    R[i] = r/sum(posEigvals)
