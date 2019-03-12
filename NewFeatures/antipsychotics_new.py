#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:04:44 2018

@author: ibarlow
"""

""" updated analysis script for antipsychotics using the new features (featuresN)
    1. loads data - no longer have to do filtering as new featuresN has means etc included
         nb. also load timeseries data as may be useful
    2. filtering based on track length and features with too many NaNs
    3. PCA on data
    4. Stats
    
    """
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import os
#import tables
import numpy as np
from tkinter import Tk, filedialog
import re

#%% import data and use file names to find out drug concentrations etc

print ('Select Data Folder')
root = Tk()
root.withdraw()
root.lift()
root.update()
root.directory = filedialog.askdirectory(title = "Select Results folder", parent = root) # show an \

#"Open" dialog box and return the path to the selected file
directory = root.directory
#find the folders within
reps = os.listdir(directory)

#now find within each rep all the feature files
fileDir ={}
#only load first three experiments
for repeat in reps:
    if repeat != '.DS_Store':
        if 'Liquid' in repeat: #filter out liquid data
            continue
        else:
            temp = os.listdir(os.path.join(directory, repeat))
            fileDir[repeat] = []
            for line in temp:
                if line.endswith('_featuresN.hdf5') == True:
                    fileDir[repeat].append(line)
    else:
        continue

#find drug names and concentrations within these file names
exp_params = {}
for rep in fileDir:
    exp_params[rep] = pd.DataFrame(data = None, columns = ['drug', 'concentration', 'date', 'uniqueID'])
    drug = []
    conc = []
    date = []
    uniqueID =[]
    for line in fileDir[rep]: #split the line and find the drug, concentration and date
        if line.split('_')[2] == 'No':
            drug.append('No_compound')
            conc.append(float(line.split('_')[4]))
            date.append(str(line.split('_') [8]))
            uniqueID.append(int(line.split('_')[-2]))
        else:
            drug.append(line.split('_')[2])
            conc.append(float(line.split('_')[3]))
            date.append(str(line.split('_') [7]))
            uniqueID.append(int(line.split('_')[-2]))
    
    exp_params[rep]['drug'] = drug
    exp_params[rep]['concentration'] = conc
    exp_params[rep]['date'] = date
    exp_params[rep]['uniqueID'] = uniqueID
    del drug, conc, date, uniqueID

#make lists of unique drugs and concentrations
uniqueDrugs = list(np.unique (exp_params[reps[1]]['drug']))
Concs = list(np.unique(exp_params[reps[1]]['concentration']))

#use re module to sort the datafiles in file directories
fileSorted ={}
for folder in fileDir:
    fileSorted[folder] = {}
    for drug in uniqueDrugs:
        fileSorted[folder][drug] =[]
        for line in fileDir[folder]:
            match = re.search(drug, line)
            if match:
                fileSorted[folder][drug].append(line)
del match, drug

#now have all the features files sorted by drug
    #loop through and load features and timeseries

features = {}
timeseries = {}
for rep in fileSorted:
    features[rep]=pd.DataFrame()
    timeseries [rep]={}
    for drug in fileSorted[rep]: #every drug
        timeseries[rep][drug] = {}
        for exp in fileSorted[rep][drug]: #every file
            with pd.HDFStore(os.path.join(directory, rep, exp),'r') as fid:
                if fid.keys()==[]:
                    continue
                else:
                    timeseries[rep][drug][exp] = fid['/timeseries_data']
                    if len(fid.groups()) <4:
                        continue
                    else:
                        #extract features means
                        temp  = fid['/features_stats'].pivot_table(columns = 'name', values = 'value')
                        temp = temp.reset_index  (drop = True)
                        #now find unique ID to append all the other information about these data
                        match = re.search ('featuresN', exp)
                        ID = int(exp[match.span()[0]-7: match.span()[0] - 1])
                        params = exp_params[rep][exp_params[rep]['uniqueID']==ID]
                        params = params[params['drug'] ==drug]
                        params = params.reset_index(drop = True)
                        temp = pd.concat([temp, pd.DataFrame(params.loc[0]).transpose()], axis =1) #add onto features 
                        
                        #and then just append to the features
                        features[rep] = features[rep].append(temp)
                        del match, params, temp
                    #remove all worm trajectories with <3000 frames, and also remove features that have too many nans
                    #refine = features[rep][drug][exp]['n_frames']>= minLength
                    #features[rep][drug][exp]= features[rep][drug][exp][refine]
    features[rep] = features[rep].reset_index(drop = True)
    del rep

#filter out features with too many nans
to_exclude  = {}
for rep in features:
    to_exclude[rep] =[]
    for feat in features[rep].columns:
        if features[rep][feat].dtype == 'O':
            continue
        else:
            n_worms = len(features[rep])
            if np.sum(np.isnan(features[rep][feat]))> 0.5*n_worms:
                to_exclude[rep].append(feat)
    to_exclude[rep] = list(np.unique(to_exclude[rep]))
del n_worms

#combined for all experiments to exclude
list_exclude = [y for v in to_exclude.values() for y in v]
list_exclude = list(np.unique(list_exclude))

#nb don't actually have any features to exclude

#%%                
"""z-score normalisation - which will be used for subsequent PCA
      use dataframe containing all the features for each experiment"""

from scipy import stats

for rep in features:
    


#first need to make a separate dataframe without the drug, concentration, data and unque ID
    #so can now do z-score normalisation without these columns
temp2 = {}
drugs = {}
concs = {}
unique = {}
date = {}
features2={}
for rep in features:
    drugs [rep] = features[rep].pop('drug')
    concs [rep] = features[rep].pop('concentration')
    date [rep] = features[rep].pop('date')
    unique [rep] = features[rep].pop ('uniqueID')
    temp2 [rep] = pd.DataFrame()
    #z-score normalisation
    for row in range(0, len(features[rep].loc[:])):
        temp = (features[rep].iloc[row,:] - \
                (np.nanmean(features[rep].iloc[:,:], axis=0)))/ \
                (np.nanstd(features[rep].iloc[:,:], axis=0))
        temp = temp.to_frame().transpose()
        temp2[rep] = temp2[rep].append(temp)
        del temp
    #add the descriptions back in
    features2[rep] = temp2[rep]
    features2[rep]['drug'] = drugs[rep]
    features2[rep]['concentration'] = concs[rep]
    features2[rep]['date'] = date[rep]
    features2[rep]['uniqueID'] = unique[rep]

del temp2

#%% 
""" PCA"""
#now to make covariance matrix for PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca = PCA
pca.fit()
#algorithm doesn't work with NaNs, so need to impute:
    #one inputation with mean, and another with median
featMatTotalNorm_mean = {}
featMatTotalNorm_med = {}
for rep in features2:
    featMatTotalNorm_mean[rep] = features2[rep].fillna(features2[rep].mean(axis = 0), inplace=True)
    featMatTotalNorm_med[rep] = features2[rep].fillna(features2[rep].median(axis = 0), inplace = True)

#just for one experiment to start with
rep = reps[0]
print (rep)

#fit and transform data onto standard scale - this means that Z-score normalising was redundant
X_std1={}
exp_names = {}
cov_mat={}
cov_mat2 ={}
for rep in features2:
    X_std1[rep] = StandardScaler().fit_transform(featMatTotalNorm_mean[rep].iloc[:,0:-4])
    #X_std2[rep] = StandardScaler().fit_transform(features2[rep].iloc[:,4:-2]) #don't include the recording info in the PCA

    exp_names [rep] = features2[rep]['date']

    cov_mat[rep] = np.cov(X_std1[rep].T)
    #cov_mat2[rep] = np.cov(X_std2[rep].T)

#function defines the pca - can actually put this earlier in the script
def pca(X_std, rep, file_type):
    """Principal Component Analysis function
    Input:
        X_std : scaled feature values
        rep : when analysiing multiple experiment as once, this is the biological replicate name
        file_type : desired output file type for figures (eg. '.tif', '.svg')
    
    Output:
        eig_vecs : eigen vectors (ie planes) for each of the principle components (type = ?)
        
        eig_vals : eigen values are the scaling factors for each eigenvector (type = )
        
        eig_pairs : tuple containg the PC eigenvector and corresponding variance explained
        
        PC_pairs : tuple containing PC number, variance explained, and cumulative variance explained
        
        PC_df: dataframe of PC_pairs
        
        cut_off : integer of the number of PCs that explain 95% of the cumulative variance
        
        PCA scree plots as images
        
        
  """
    eig_vecs, eig_vals, v, = np.linalg.svd(X_std.T)
    #test the eig_vecs
    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print ('Everything OK!')
    
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    #then sort - high to low
    eig_pairs.sort(key =lambda tup:tup[0])
    eig_pairs.reverse()

    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])
    
    #make plots
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp= np.cumsum(var_exp)
    #add in cut off for where 95% variance explained
    cut_off = cum_var_exp <95
    cut_off = np.argwhere(cut_off)
    cut_off = int(cut_off[-1])

    #first make dataframe with all the PCs in
    x=['PC %s' %i for i in range(1,len(eig_vals))]
    y= var_exp[0:len(eig_vals)]
    z=cum_var_exp[0:len(eig_vals)]
    PC_pairs= [(x[i], y[i], z[i]) for i in range(0,len(eig_vals)-1)]
    PC_df = pd.DataFrame(data=PC_pairs, columns = ['PC', 'variance_explained', \
                                                'cum_variance_explained'])
    sns.set_style ('whitegrid')
    f, (ax1,ax2) = plt.subplots(ncols=2, sharey=True)
    plt.title(rep)
    trace1 = sns.barplot(y= 'variance_explained', x= 'PC', data=PC_df, ax=ax1)
    sns.despine()
    ax1.xaxis.set_ticks(np.arange(0,70,10))
    ax1.xaxis.set_ticklabels(PC_df['PC'][0:71:10])
    ax1.axes.tick_params(axis = 'x', rotation = 45, direction = 'in', labelbottom = True)
    ax1.xaxis.axes.set_xlim(left = 0, right= 70)
    trace2 = sns.stripplot(y='cum_variance_explained', x='PC', data=PC_df, ax=ax2)
    ax2.xaxis.set_ticks(np.arange(0,70,10))
    ax2.xaxis.set_ticklabels(PC_df['PC'][0:71:10])
    ax2.axes.tick_params(axis = 'x', rotation = 45, direction = 'in', labelbottom = True)
    ax2.xaxis.axes.set_xlim(left = 0, right= 70)
    trace2 = plt.plot([cut_off, cut_off], [0,95], linewidth =2)
    plt.text(cut_off, 100, str(cut_off))
    
    sns.despine()
    f.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_PC_variance_explained.' + file_type), dpi=400)
    plt.show()
    del x,y, z, tot, var_exp, cum_var_exp, f, ax1, ax2, trace1, trace2
    
    return eig_vecs, eig_vals, eig_pairs, PC_pairs, PC_df, cut_off

eig_vecs1={}
eig_vals1 = {}
eig_pairs1 = {}
PC_pairs1={}
PC_df1 = {}
cut_off1={}

for rep in X_std1:
    eig_vecs1[rep], eig_vals1[rep], eig_pairs1[rep], PC_pairs1[rep],\
    PC_df1[rep], cut_off1[rep] = pca(X_std1[rep], rep)
      



