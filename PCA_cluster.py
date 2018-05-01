#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:24:14 2018

@author: ibarlow
"""

""" script to do PCA on entire antipsychotics dataset (agar only) and then
Dcluster on the top x PCs that explain 95% of the variance. Use mahalanobis
distance  in the Dcluster"""

import TierPsyInput as TP
import numpy as np
import pandas as pd

directoryA, fileDirA, featuresA =  TP.TierPsyInput('new', 'Liquid')

#from here pull out info about each of the experiment's variables, such as drug, concentration and uniqueID

    #first, remove experiment list
exp_namesA={}
featuresA2 = featuresA.copy()
for rep in featuresA:
    exp_namesA[rep] = featuresA[rep].pop('exp')

drugA = {}
concA = {}
dateA = {}
uniqueIDA = {}

for rep in exp_namesA:
    drugA[rep], concA[rep], dateA[rep], uniqueIDA[rep] = TP.extractVars(exp_namesA[rep])


#make lists of unqiue drugs and concs
drugs = []
concs = []
dates =[]
for rep in drugA:
    drugs.append(list(drugA[rep].values))
    concs.append(list(concA[rep].values))
    dates.append(list(dateA[rep].values))
flatten = lambda LoL: [item for sublist in LoL for item in sublist] #lambda function flattens list of lists (LoL)

uniqueDrugs = list(np.unique(flatten(drugs)))
uniqueConcs = list(np.unique(flatten(concs)))
uniqueDates = list(np.unique(flatten(dates)))
del drugs, concs, dates

#%% import eggs data
import os

#fileid for eggs data
fid_eggs = os.path.join (directoryA[:-7], 'ExtraFiles', 'egg_all.csv')
#load data
eggs_df = pd.read_csv(fid_eggs)

#split the data by experiment
eggs={}
reps = list(fileDirA.keys())

#make dictionary to match rep and date
rep_match = {}
for rep in dateA:
    rep_match[list(np.unique(dateA[rep].values))[0]] = rep

#and then compile dictionary using this   
for date in uniqueDates:
    eggs[rep_match[date]] = eggs_df[eggs_df['date'] ==int(date)]
    eggs[rep_match[date]] = eggs[rep_match[date]].reset_index(drop=True)

del rep_match, reps
#problem is that there is not egg data for every plate... so need to match up data accordingly

#add on descriptors to A2 dataframe
featuresA3 = {}
for rep in featuresA2:
    featuresA3[rep] = pd.concat([featuresA2[rep], dateA[rep], drugA[rep], concA[rep], uniqueIDA[rep]], axis =1)

#now to match them up
eggs2 = eggs.copy()
eggs_df2 = {}
for rep in featuresA3:
    eggs_df2[rep] = pd.DataFrame()
    for step in range(0, featuresA3[rep].shape[0]):
        line = featuresA3[rep].iloc[step,:]
        temp = eggs[rep][eggs[rep]['uniqueID'] == line['uniqueID']]
        if temp.shape[0]>0:
            temp2 = temp[temp['drug'] == line['drug']]
            temp2 = temp2[temp2['concentration'] == line['concentration']]
            if temp2.shape[0] == 0:
                temp3 = line.copy()
                temp3['total'] = float('nan')
            else:
                if temp2.shape[0]>1:
                    temp3 = temp2.iloc[0,:] #this is so that only place the data for one if there are duplicates
                    eggs[rep] = eggs[rep].drop(temp3.to_frame().transpose().index)
                else:
                    temp3 = temp2
            del temp2
        else:
            #put nan in place
            temp3 = line.copy()
            temp3['total'] = float('nan')
            
            
        line = line.to_frame().transpose()   
        line['eggs'] = float(temp3['total'])
        
        eggs_df2[rep] = eggs_df2[rep].append(line)
        del line, temp, temp3

#Call this new dataframe featuresEA_1
featuresEA_1 = eggs_df2.copy()
    
del eggs2,featuresA3

#need to pop out the experiment descriptors again as may be wrong
drugA2 = {}
concA2 ={}
dateA2={}
uniqueIDA2 = {}

for rep in featuresEA_1:
    drugA2[rep] = featuresEA_1[rep].pop('drug')
    concA2[rep] = featuresEA_1[rep].pop('concentration')
    dateA2[rep] = featuresEA_1[rep].pop('date')
    uniqueIDA2[rep] = featuresEA_1[rep].pop('uniqueID')
    
#%% Z-score normalisation

featuresZ={}
for rep in featuresA:
    featuresZ[rep] = TP.z_score(featuresEA_1[rep])

to_excludeA={}
for rep in featuresZ:
    to_excludeA[rep] = TP.FeatFilter(featuresZ[rep])
    
#combined for all experiments to exclude
list_exclude = [y for v in to_excludeA.values() for y in v]
list_exclude = list(np.unique(list_exclude))

#remove these features from the features dataframes
featuresZ1 = {}
for rep in featuresZ:
    featuresZ1[rep] = TP.FeatRemove(featuresZ[rep], list_exclude)

#%% 
    
#data normalization
#combine the Z-scored data as in the tSNE - this is better for when combining all the experiments
     #first impute the nans in the Z-scored data

featMatTotalNorm_mean = {}
for rep in featuresZ1:
    featMatTotalNorm_mean[rep] = featuresZ1[rep].fillna(featuresZ1[rep].mean(axis = 0))
    
featuresZ2 = {}
featMatAll = pd.DataFrame()
for rep in featMatTotalNorm_mean:
    featuresZ2 [rep] = pd.concat([featMatTotalNorm_mean[rep], drugA2[rep], concA2[rep], dateA2[rep]], axis =1)
    featMatAll = featMatAll.append(featuresZ2[rep])

#reset index
featMatAll = featMatAll.reset_index(drop = True)
featMatAll2 = featMatAll.copy()

drug_all = featMatAll2.pop ('drug')
conc_all = featMatAll2.pop('concentration')
date_all = featMatAll2.pop ('date')

reps = list(featuresZ2.keys())

#%% so now onto the PCA

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#make array of z-scored data
X = np.array(featMatAll2.values)
pca = PCA()
X2= pca.fit_transform(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)
thresh = cumvar <= 0.95 #set 95% variance threshold
cut_off = int(np.argwhere(thresh)[-1])

#make a plot
sns.set_style('whitegrid')
plt.plot(range(0, len(cumvar)), cumvar)
plt.plot([cut_off, cut_off], [0, 1], 'k')

#now put the 1:cut_off PCs into a dataframe
PCname = ['PC_%d' %(p+1) for p in range (0,cut_off+1)]
PC_df = pd.DataFrame(data= X2[:,:cut_off+1], columns = PCname)
PC_df['drug'] = drug_all
PC_df['concentration'] = conc_all
PC_df['date'] = date_all

#%%
#maybe I should scale the PCs before doing the clustering??

#just try using Dcluster package
from scipy import spatial
import Dcluster as dcl
import os
import density_plots as dp

#calcualte euclidean distance for PCdata
temp = spatial.distance.pdist(PC_df.iloc[:,1:2], metric='euclidean')
temp = spatial.distance.squareform(temp)

#this make squareform distance matrix so assume NDx == NDy

#save temp as a .txt file
fileid1 = os.path.join(directoryA[0:-7], 'pca_pdist.txt')
np.savetxt(fileid1, temp, delimiter = '\t', fmt = '%f')

#also save the file identifiers
fileid2 = os.path.join(directoryA[0:-7], 'pca_new_ids.csv')
PC_df.iloc[:,-3:].to_csv(fileid2)

#load file
dist1 = np.genfromtxt(fileid1, delimiter = '\t')
ids = pd.read_csv(fileid2, delimiter = ',').drop(['Unnamed: 0'],axis=1)

#vectorise dist values
xxdist1 = [(dist1[t2].T) for t2 in range(0,len(dist1))]
xxdist1 = np.concatenate(xxdist1[:])
ND1 = dist1.shape[0]
N1 = xxdist1.shape[0]

#can use this squareform pdist matrix in dcluster - will be alot faster and save on memory as well
(XY1, S1) = dcl.mds(dist1)

#define rho-delta values
(rho1, delta1, ordrho1,dc1, nneigh1) = dcl.rhodelta(dist1, xxdist1, ND1, N1)

#plot
sns.set_style('whitegrid')
f,ax, rho_d = dp.plot1(rho1, delta1)
f.savefig(os.path.join(directoryA[0:-7], 'Figures', 'pca_decision_graph.png'), dpi =200)

   
#based on these plots set rhomin and delta min
rhomin1 = 2
deltamin1 = rho_d/2

#plot the decision map
ax, clusters1 = dp.density_calc (dist1, XY1, ND1, rho1, delta1,ordrho1,dc1,nneigh1,rhomin1, deltamin1, directoryA[0:-7])
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'pca_decision_graph2.png'), dpi =200)

#make a dataframe
dc_df = pd.DataFrame (data = XY1, columns = ['x', 'y'])
dc_df['cluster'] = clusters1[:,1]
dc_df['cluster_halo'] = clusters1[:,2]
dc_df['drug'] =ids['drug']
dc_df['concentration'] = ids['concentration']
N_clusters = int(max(clusters1[:,1]))

#make the plots
sns.set_palette(sns.color_palette("tab20", N_clusters))
sns.lmplot(x= 'x', y='y', data=dc_df, hue = 'cluster', fit_reg = False, legend=False)
plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
plt.tight_layout(rect=[0,0,0.75,1])
plt.ylim[-1, 1]
#plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'clusters__only_new.png'), dpi = 200)

sns.set_palette(sns.color_palette("tab20", N_clusters))
sns.lmplot(x= 'x', y='y', data=dc_df, hue = 'drug', fit_reg = False, legend=False)
plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
plt.tight_layout(rect=[0,0,0.75,1])
plt.ylim([-1, 1])
#plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'drugs_clusters_new.png'), dpi = 200, bbox_inches = 'tight')
