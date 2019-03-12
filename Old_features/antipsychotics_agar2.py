#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:14:29 2018

@author: ibarlow
"""
"""antispyschotics analysis using only the experiments done on agar - ie the first 
three. This probably makes a difference for the first filtering stage."""

import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import os
#import tables
import numpy as np
from tkinter import Tk, filedialog
import re

#%% first need to import the drug desciptions and data

root = Tk()
root.withdraw()
root.lift()
root.update()
root.directory = filedialog.askopenfilename (title = 'select list', parent = root)
root.destroy()

exp_describe = root.directory

del root

with open(exp_describe, 'r') as fid:
    textData = fid.readlines()
    
    #remove newline characters
    for line in range(0,len(textData)):
        textData[line] = textData[line].rstrip()

#now create separate lists of file, drugs and concentrations
fileNames = []
drugNames = []
Conc = []
for line in textData:
    fileNames.append(line.split(',')[0])
    drugNames.append(line.split(',')[1])
    Conc.append(line.split(',')[2])

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
                if line.endswith('_features.hdf5') == True:
                    fileDir[repeat].append(line)
    else:
        continue
            
#need to match up the compounds with those from the drug list
uniqueDrugs = list(np.unique(drugNames))
uniqueDrugs.append('No_Compound')

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
    #loop through and load features timeseries
#now loop through the files and only load the feature files

#minimum trajectory length                
minLength = 3000;

features = {}
for rep in fileSorted:
    features[rep]={}
    for drug in fileSorted[rep]:
        features[rep][drug]={}
        for exp in fileSorted[rep][drug]: 
            with pd.HDFStore(os.path.join(directory, rep, exp),'r') as fid:
                if fid.keys()==[]:
                    continue
                else:
                    features[rep][drug][exp] = fid['/features_summary/means']
                    #remove all worm trajectories with <3000 frames, and also remove features that have too many nans
                    refine = features[rep][drug][exp]['n_frames']>= minLength
                    features[rep][drug][exp]= features[rep][drug][exp][refine]
del refine, rep

#filter out features with too many nans
to_exclude  = {}
for rep in features:
    to_exclude[rep] =[]
    for drug in features[rep]:
        for exp in features[rep][drug]:
            for feat in features[rep][drug][exp]:
                n_worms = len(features[rep][drug][exp])
                if np.sum(np.isnan(features[rep][drug][exp][feat]))> 0.5*n_worms:
                    to_exclude[rep].append(feat)
    to_exclude[rep] = list(np.unique(to_exclude[rep]))

#combined for all experiments to exclude
list_exclude = [y for v in to_exclude.values() for y in v]
list_exclude = list(np.unique(list_exclude))

#original features saved as copy just in case:
features_copy = features

for rep in features:
    for drug in features[rep]:
        for exp in features[rep][drug]:
                features[rep][drug][exp].drop(list_exclude, axis = 1, inplace= True)

#now features only contains filtered features; features_copy is old features

#concatenate dataframes from each experiment to contain data for each worm - 
                #- with drug and dose
features3 = {}
for rep in features:
    features3[rep] = pd.DataFrame()
    for drug in features[rep]:
        concs = list(features[rep][drug].keys())
        for line in concs:
            match = re.search(drug, line)
            conc1 = float(re.search(r'\d+', line[match.end()+1:match.end()+4]).group())
            temp = features[rep][drug][line]
            temp['drug'] = drug
            temp['concentration'] = conc1
            temp['date'] = line
            features3[rep]= features3[rep].append(temp)
            del temp, match, conc1
    features3[rep].reset_index(drop = True)

#now have a dataframe that conctains all the data for each rep in one DataFrame   
    
#%%                
#z-score normalisation
    #first create a summary of the plate by taking the average
      #then compile this into one dataframe - all drugs and all doses
        #then can do the calculation

#create new dataframe containing the means
temp2 ={}
for rep in features:
    temp2[rep] = pd.DataFrame()
    for drug in features[rep]:
        if features[rep][drug] =={}:
            continue
        for exp in features[rep][drug]:
            temp = features[rep][drug][exp].mean(skipna=True)
            temp = temp.to_frame().transpose()
            temp['experiment'] = exp
            #add in the concentrations as an extra column
            if drug == 'DMSO':
                temp['concentration'] = float(10)
                temp['drug'] = drug
            else:
                match = re.search(drug, exp)
                if match:
                    conc = float(re.search(r'\d+', exp[match.end()+1:match.end()+4]).group())
                    temp['concentration']= conc
                    temp ['drug'] = drug
                    del conc
            temp2[rep] = temp2[rep].append(temp)
            del temp, exp

#update in new features frame
features2 = {}
for rep in features:
    features2[rep] =[]
    features2[rep] = temp2[rep].reset_index(drop=True)

del temp2

#feat matrix normalisation
temp2 ={}
for rep in features2:
    temp2 [rep] = pd.DataFrame()
    for row in range(0, len(features2[rep].loc[:])):
        temp = (features2[rep].iloc[row,0:-3] - \
                (np.nanmean(features2[rep].iloc[:,0:-3], axis=0)))/ \
                (np.nanstd(features2[rep].iloc[:,0:-3], axis=0))
        temp = temp.to_frame().transpose()
        temp2[rep] = temp2[rep].append(temp)
        del temp

for rep in features2:
    temp2[rep]['experiment'] = features2[rep]['experiment']
    temp2[rep]['concentration'] = features2[rep]['concentration']
    temp2[rep]['drug'] = features2[rep]['drug']
    
featMatTotalNorm = temp2
del temp2

#%%
#now to make covariance matrix for PCA
from sklearn.preprocessing import StandardScaler

#algorithm doesn't work with NaNs, so need to impute:
    #one inputation with mean, and another with median
featMatTotalNorm_mean = {}
featMatTotalNorm_med = {}
for rep in featMatTotalNorm:
    featMatTotalNorm_mean[rep] = featMatTotalNorm[rep].fillna(featMatTotalNorm[rep].mean(axis = 0), inplace=True)
    featMatTotalNorm_med[rep] = featMatTotalNorm[rep].fillna(featMatTotalNorm[rep].median(axis = 0), inplace = True)

#just for one experiment to start with
rep = reps[0]
print (rep)

#fit and transform data onto standard scale - this means that Z-score normalising was redundant
X_std1={}
#X_std2={}
exp_names = {}
cov_mat={}
cov_mat2 ={}
for rep in features2:
    X_std1[rep] = StandardScaler().fit_transform(featMatTotalNorm_mean[rep].iloc[:,4:-3])
    #X_std2[rep] = StandardScaler().fit_transform(features2[rep].iloc[:,4:-2]) #don't include the recording info in the PCA

    exp_names [rep] = features2[rep]['experiment']

    cov_mat[rep] = np.cov(X_std1[rep].T)
    #cov_mat2[rep] = np.cov(X_std2[rep].T)

#function defines the pca - can actually put this earlier in the script
def pca(X_std, rep, file_type):
    """pca function that returns PCA scree plots and ..."""
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

#now to find the top features that contribute to PC1 and PC2
def PC_feats(eig_pairs3, cut_offs, features_2):
    """ finds the top features and returns dataframes with contributionsa nd features"""
    x = ['PC_%s' %i for i in range(1,cut_offs+1)]
    PC_contribs = [(eig_pairs3[i][1]) for i in range (0,cut_offs)]
    features_1 = list(features_2.columns[4::1])
    PC_features = pd.DataFrame(PC_contribs)
    PC_features = PC_features.T
    PC_features.columns = x
    PC_features['features'] = features_1[0:-3]
    
    #rank the features
    PC_tops = {}
    for PC in PC_features.columns:
        PC_tops[PC] = list(PC_features[PC].sort_values().index[:])
        PC_tops[PC].reverse()
        PC_tops[PC] = PC_features['features'].iloc[PC_tops[PC]]
    return PC_contribs, PC_features, PC_tops, x

PC_conts1 = {}
PC_feats1 = {}
PC_top1={}
x1 ={}
for rep in eig_pairs1:
    PC_conts1[rep], PC_feats1[rep], \
    PC_top1[rep], x1[rep] = PC_feats(eig_pairs1[rep], cut_off1[rep], features2[rep])

#biplot function
def biplot(ranks, coeff, pc1, pc2, n_feats, rep, file_type):
    """ biplot function  - specify output file type"""
    cmap = sns.color_palette("husl", len(uniqueDrugs))
    sns.set_style('whitegrid')
    pcs = ('PC_%d' %(pc1), 'PC_%d' %(pc2))
    for pc in range(len(pcs)):
        if pc == 1:
            for i in range (n_feats):
                plt.arrow(0,0,\
                          coeff[np.flip(pcs,axis=0)[pc]].iloc[ranks[pcs[pc]].index[i]],  \
                          coeff[pcs[pc]].iloc[ranks[pcs[pc]].index[i]],\
                          color = cmap[pc], alpha = 1, label = pcs[pc])    
                if coeff is None:
                    continue
                else:
                    plt.text (coeff[np.flip(pcs,axis =0)[pc]].iloc[ranks[pcs[pc]].index[i]]*3, \
                                    coeff[pcs[pc]].iloc[ranks[pcs[pc]].index[i]]*1.5, \
                              coeff['features'].iloc[ranks[pcs[pc]].index[i]], color = cmap[pc],\
                              ha = 'center', va='center')
        else:
            for i in range (n_feats):
                plt.arrow(0,0, coeff[pcs[pc]].iloc[ranks[pcs[pc]].index[i]],\
                          coeff[np.flip(pcs,axis=0)[pc]].iloc[ranks[pcs[pc]].index[i]],\
                          color = cmap[pc], alpha = 1, label = pcs[pc])    
                if coeff is None:
                    continue
                else:
                    plt.text (coeff[pcs[pc]].iloc[ranks[pcs[pc]].index[i]]*4, \
                              coeff[np.flip(pcs,axis =0)[pc]].iloc[ranks[pcs[pc]].index[i]]*3,\
                              coeff['features'].iloc[ranks[pcs[pc]].index[i]], color = cmap[pc],\
                              ha = 'center', va='center')
    plt.xlim (-1, 1)
    plt.ylim (-1,1)
    #plt.axis('equal')
    plt.xlabel ('PC_1')
    plt.ylabel('PC_2')
    plt.legend()
    plt.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_biplot.' + file_type), dpi =200)
    plt.show()

#now make biplots for all the reps 
for rep in PC_top1:
    biplot(PC_top1[rep], PC_feats1[rep],1,2, 1, rep, 'svg')

#%% now to transform into feature space
    #concanenate the eigen_vector matrix across the top 80 eigenvalues

def feature_space(features_1, eig_pairs, X_std, cut_offs, x):
    """ input = original feature matrix (after filtering), eigenpairs, X_std matrix, 
    cut-off values and the vector containing the PC names"""
    matrix_w = eig_pairs[0][1].reshape(eig_pairs[0][1].size,1)
    for i in range(1,cut_offs):
        temp_matrix = eig_pairs[i][1].reshape(eig_pairs[i][1].size,1)
        matrix_w = np.hstack((matrix_w, temp_matrix))
        del temp_matrix
    print ('Matrix W: \n', matrix_w)
    
    Y = X_std.dot(matrix_w)
    PC_df = pd.DataFrame(Y)
    PC_df.columns = x
    PC_df['drug'] = features_1['drug']
    PC_df['concentration'] = features_1['concentration']
    PC_df['experiment'] = features_1['experiment']
    return matrix_w, Y, PC_df

matrix_w1 = {}
Y1 = {}
PC_df2 = {}
for rep in features2:
    matrix_w1[rep], Y1[rep], PC_df2[rep] = feature_space(features2[rep], eig_pairs1[rep],\
            X_std1[rep], cut_off1[rep], x1[rep])

#to make plots    
def PC12_plots (df, dose, rep, file_type):
    """this makes plots that are scaled PCs"""
    sns.set_style('whitegrid')
    cmap = sns.choose_colorbrewer_palette(data_type = 'q')
    if dose == []:
        temp = df
    else:
        to_plot = (df['concentration'] == float(dose))# or (df['concentration'] == float(14))
        temp = df[to_plot]
        temp = temp.append(df[df['drug']=='DMSO']) #add on DMSO controls
        temp = temp.append (df[df['drug'] == 'No_Compound'])
    xs = temp['PC_1']
    ys = temp['PC_2']
    scalex = 1/(xs.max() - xs.min())
    scaley = 1/(ys.max() - ys.min())
    temp ['PC_1'] = temp['PC_1'].replace(temp['PC_1'].values, xs*scalex)
    temp['PC_2'] = temp['PC_2'].replace(temp['PC_2'].values, ys*scaley)
    f = plt.figure
    f= sns.lmplot(x= 'PC_1', y='PC_2', data= temp, hue = 'drug',fit_reg = False)
    #add in arrow plot
    #f = plt.arrow(0,0, PC_feats['PC_1'].iloc[PC_top['PC_1'].index[0]],\
         #             PC_feats['PC_2'].iloc[PC_top['PC_1'].index[i]], \
          #            color = 'm', alpha = 1, linewidth = 2)
    
    plt.xlim (-1, 1)
    plt.ylim (-1,1)
    plt.title ('concentration = ' + str(dose))
    plt.savefig (os.path.join(directory[0:-7], 'Figures', rep + '_'\
                              + str(dose) + '_PC12_norm.' + file_type), dpi = 200)

#now make the plots   
for rep in PC_df2:
    for i in [1,10,100,200]:
        PC12_plots(PC_df2[rep], i, rep, 'tif')
    
#now can make dataframe containing means and column names to plot trajectories through PC space
def PC_av(PC_dataframe, x):
    """function to convert to average PC for replicates. Requires PC dataframe
    and x containing all the column name"""
    PC_means= pd.DataFrame(data = None, columns = x)
    uniqueDrugs1 = np.unique(PC_dataframe['drug'])
    for drug in uniqueDrugs1:
        finders = PC_dataframe['drug'] == drug
        keepers = PC_dataframe[finders]
        concs = plt.unique(keepers['concentration'])
        for dose in concs:
            refine = keepers['concentration'] == dose
            final = keepers[refine]
            temp = final.iloc[:,0:-2].mean(axis=0)
            temp = temp.to_frame().transpose()
            temp['drug'] = drug
            temp['concentration'] = dose
            PC_means= PC_means.append(temp)
            del refine, final, temp
        del finders, keepers, concs

    PC_means = PC_means.reset_index(drop=True)
    return PC_means

PC_means1={}
for rep in PC_df2:
    PC_means1[rep] = PC_av(PC_df2[rep], x1[rep])
    

def PC_traj(df,rep, file_type):
    """this function groups by drug an plots the trajectories through PC space""" 
    #scale the PCs
    xscale = 1/(np.max(df['PC_1']) - np.min(df['PC_1']))
    yscale = 1/(np.max(df['PC_2']) - np.min(df['PC_2']))
    #okay so now have a summary of each drug for each PC.
        #scale and plot the drugs across the PC1 and 2 space
    uniqueDrugs1 = np.unique(df['drug'])
    cmap = sns.choose_colorbrewer_palette(data_type = 'q')
    #cmap = sns.color_palette("husl", len(uniqueDrugs1)) #set colormap
    for drug in range(len(uniqueDrugs1)):
        to_plot = df['drug'] == uniqueDrugs1[drug]
        plotting1 = df[to_plot]
        ax = plt.plot(plotting1['PC_1']*xscale, plotting1['PC_2']*yscale, \
                      linewidth =2, color = cmap[drug], marker = 'o', \
                      label = uniqueDrugs1[drug])
    plt.xlim (-1,1)
    plt.ylim(-1,1)
    plt.legend(loc = 2,ncol = 3, frameon= True)
    plt.xlabel ('PC_1')
    plt.ylabel('PC_2')
    plt.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_PCtraj.' + file_type), dpi = 200)
    plt.show()

sns.set_style('whitegrid')
for rep in PC_means1:
    PC_traj(PC_means1[rep], rep, 'svg')

#%% now to PCA on all experiments combined
    #use averages for each drug and each dose - from the technical replicates
        #so first need to take averages from FeatMatNorm
            #and then concatenate to combine
                #then do PCA

featMatNorm2 = pd.DataFrame()
for rep in featMatTotalNorm:
    concs = np.unique(featMatTotalNorm[rep]['concentration'])
    for drug in uniqueDrugs:
        test = featMatTotalNorm[rep]['drug'] == drug
        drug_all = featMatTotalNorm[rep][test]
        for conc in concs:
            test2 = drug_all['concentration']==conc
            if np.sum(test2) == 0:
                continue
            else:
                drug_concs = drug_all[test2]
                mean1 = drug_concs.mean(axis=0).to_frame().transpose()
                mean1['drug'] = drug
                mean1['concentration'] = conc
                mean1['experiment'] = rep
                featMatNorm2 = featMatNorm2.append(mean1)
                del mean1, test2, drug_concs
        del test, drug_all            
featMatNorm2 = featMatNorm2.reset_index(drop=True)

#now I have the space for doing PCA
    #however there are nans in the dataset (due to the liquid dataset)
        #1. remove the liquids and do PCA
        #2. keep liquids and interpolate the means

featMatNorm4 = featMatNorm2.fillna(featMatNorm2.mean(axis = 0), inplace=True)

#standard scalar before doing the the PCA
X_std_combi  = StandardScaler().fit_transform(featMatNorm4.iloc[:,4:-3])

#PCA function
eig_vecs_combi, eig_vals_combi, eig_pairs_combi,\
PC_pairs_combi, PC_df_combi, cut_off_combi = pca(X_std_combi, 'agar', 'tif')

#eig_pairs and biplots
PC_conts_combi, PC_feats_combi, PC_top_combi, x_combi = PC_feats(eig_pairs_combi, cut_off_combi, featMatNorm2)

#make the biplots
biplot(PC_top_combi, PC_feats_combi,1,2, 1, 'agar', 'svg')

#transform into feature space   
matrix_w_combi,  Y_combi, PC_df_combi2 = feature_space(featMatNorm4,\
              eig_pairs_combi, X_std_combi, cut_off_combi, x_combi)


#now to make plots
for i in [1,10,100,200]:
    PC12_plots(PC_df_combi2, i, 'agar')

PC12_plots(PC_df_combi2, [], 'agar', 'tif')

PC_means_combi = PC_av(PC_df_combi2, x_combi)
   
PC_traj(PC_means_combi, 'agar', 'svg')


#%% now to do some stats
from scipy import stats

#compare each compound to control data
controlMeans = {}
for rep in features2:
    controlMeans[rep] = pd.DataFrame()
    for line in range(len(features2[rep])):
        if 'DMSO' in features2[rep]['experiment'].iloc[line]:
            DMSO = features2[rep].iloc[line]
            controlMeans[rep] = controlMeans[rep].append(DMSO.to_frame().transpose())
            del DMSO
    controlMeans[rep].reset_index(drop=True)

#so can now compile a matrix of p_values comparing each drug to control means
    #need to loop through the different drugs and basically just compare all the features to controlMeans
pVals = {}
feats =list(features2[rep].columns)
feats = feats[4:-3]
feats.append('drug')
feats.append('concentration')
for rep in features2:
    pVals[rep] = pd.DataFrame()
    for drug in uniqueDrugs:
        if drug == 'DMSO':
            continue
        else:        
            currentInds = pd.DataFrame()
            for line in features2[rep].index:
                match = re.search(drug, features2[rep]['experiment'].iloc[line])
                if match:
                    current = features2[rep].iloc[line].to_frame().transpose()
                    currentInds = currentInds.append(current)
                    del current
            
            currentInds.reset_index(drop=True)
            
            #test if dataframe is empty to continue to next drug
            if currentInds.empty:
                continue
            else:
                #separate the concentrations
                conc= np.unique(currentInds['concentration'])
                for dose in conc:
                    test =[]
                    to_test = currentInds['concentration'] ==dose
                    testing = currentInds[to_test]
                    for feature in currentInds.columns[4:-3]:
                        test.append(stats.ttest_ind(testing[feature], controlMeans[rep][feature]))
       
                    ps = [(test[i][1]) for i in range(len(test))] #make into a list
                    ps.append(drug)
                    ps.append(dose)
        
                    temp = pd.DataFrame(ps).transpose()
                    pVals[rep] = pVals[rep].append(temp)
                    del temp, to_test, testing
            del currentInds, match

    #add in features
    pVals[rep].columns = feats
    pVals[rep] = pVals[rep].reset_index (drop=True)   

#import module for multiple comparison correction
import statsmodels.stats.multitest as smm

#now correct for multiple comparisons
bh_p={}
top_feats = {}
post_exclude = {}
sig_feats = {}
for rep in pVals:
    bh_p [rep] = pd.DataFrame()
    for cond in range(pVals[rep].shape[0]):
        reg, corrP, t,s =  smm.multipletests(pVals[rep].values[cond, :-2],\
                                             alpha=0.05, method = 'fdr_bh', \
                                             is_sorted = False, returnsorted = False)
        corrP = list(corrP)
        corrP.append(pVals[rep]['drug'].iloc[cond])
        corrP.append(pVals[rep]['concentration'].iloc[cond])
        corrP = pd.DataFrame(corrP).transpose()
        bh_p [rep]= bh_p[rep].append(corrP)
        del corrP
        #add in the feature names
    bh_p[rep].columns = feats
    bh_p [rep]= bh_p[rep].reset_index(drop = True)

    #now can filter out the features that have no significant p-values
    top_feats[rep]= bh_p[rep].values[:,0:-2] <=0.05
    post_exclude [rep]= []
    sig_feats[rep] = []
    for feat in range(bh_p[rep].shape[1]-2):
        if np.sum(top_feats[rep][:,feat]) ==0:
            post_exclude[rep].append(bh_p[rep].columns[feat])
        else:
            sig_feats[rep].append((bh_p[rep].columns [feat], np.sum(top_feats[rep][:,feat])))
    
    #sort by most to least
    sig_feats[rep].sort(key =lambda tup:tup[1])
    sig_feats[rep].reverse()


#make some violin plots of the significant features

def swarms (rep1, feature1):
    cmap = sns.choose_colorbrewer_palette(data_type = 'q')
    sns.swarmplot(x='drug', y=feature1, data=features3[rep1], \
                      hue = 'concentration', palette = cmap)
    plt.xticks(rotation = 45)
    plt.savefig(os.path.join (directory[0:-7], 'Figures', rep1 + feature1 + '.svg'), dpi = 200) 

swarms (reps[2], 'midbody_speed_abs')
swarms (reps[0], 'head_bend_mean')



#attempt at clustergrid
cg = sns.clustermap(featMatTotalNorm[rep].iloc[:,4:-3], col_cluster = True, standard_scale =1,\
                    metric = 'euclidean')
cg.ax_heatmap.set_yticklabels(featMatTotalNorm[rep]['drug'][cg.dendrogram_row.reordered_ind])
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
