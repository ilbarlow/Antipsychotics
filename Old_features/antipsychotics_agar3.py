#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:13:38 2018

@author: ibarlow
"""

""" updated script for running the old features using Agar data only"""
#%%

from tkinter import Tk, filedialog
import pandas as pd
import os

def TierPsyInput(version, exclude):
    """ A filedialog will open to select the results folder. The script will 
    then search through this folder for the features files
    
    Input:
        version - 'old' or 'new'
                This determines which features are imported
        
        exclude - the name of any folders to exclude
        
    Output:
        directory - pathname of the selected results folder
        
        features - a dictionary containing the features timeseries (old) or
            summaries (new) for each results folder"""
    
    print ('Select Data Folder')
    root = Tk()
    root.withdraw()
    root.lift()
    root.update()
    root.directory = filedialog.askdirectory(title = "Select Results folder", \
                                             parent = root)
    
    if root.directory == []:
        print ('No folder selected')
    else:
        directory = root.directory
        #find the folders within
        reps = os.listdir(directory)
        
        #now to test version
        if version == 'new':
            feat_file = '_featuresN.hdf5'
        elif version == 'old':
            feat_file = '_features.hdf5'
        else:
            print ('Version not specified!')
        
        #now find within each subfolder all the feature files
        fileDir ={}
        for repeat in reps:
            if repeat != '.DS_Store': #ignore these hidden files  
                if exclude in repeat: #filter out data to exclude
                    continue
                else:
                    temp = os.listdir(os.path.join(directory, repeat))
                    fileDir[repeat] = []
                    for line in temp:
                        if line.endswith(feat_file) == True:
                            fileDir[repeat].append(line)
                        else:
                            continue
        
        #now have a dictionary of all the filenames to load
            #can now load them
        
        features ={}
        for rep in fileDir:
            features[rep] = pd.DataFrame()
            for line in fileDir[rep]:
                with pd.HDFStore(os.path.join(directory, rep, line), 'r') as fid:
                    if version == 'old':
                        temp = fid['/features_summary/means']
                    elif version == 'new':
                        if len(fid.groups()) <4:
                            continue
                        else:
                            temp = fid['/features_stats'].pivot_table(columns = 'name', values = 'value')
                    
                    temp['exp'] = line
                    temp = temp.reset_index  (drop = True)
                    features[rep] = features[rep].append(temp)
                    del temp
            features[rep] = features[rep].reset_index(drop=True)
    
    return directory, fileDir, features

#now call these functions
directoryA, fileDirA, featuresA = TierPsyInput('old', 'Liquid')

#%%
#now for filtering of tracks and features

import numpy as np

def TierFilter(features, minLength):
    """ the function removes tracks with fewer frames than the set threshold,
    and also tracks with more than >50% nans
    
    Input:
        features - the features dataframe
        minLength - the threshold length (eg. 3000)
        
    Output:
        features - the features dataframe with tracks removed"""
    #first refine   
    refine = features['n_frames']>= minLength
    features= features[refine]
    
    #second refine - remove tracks with too many nans
    refine2 = features.count(axis=1) > (features.shape[1]/2)
    features = features[refine2]
    
    return features

for rep in featuresA:
    featuresA[rep] = TierFilter(featuresA[rep], 3000)

#and then filter based on features with >50% nans
def FeatFilter(features):
    """ this function removes features with too many NaNs
    Input:
        features - dataframe of features
    
    Output:
        to_exclude - list of features with >50% NaNs"""
    
    to_exclude  = []
    n_worms = features.shape[0]
    for feat in features.columns:
        if features[feat].dtype == object:#exclude experiment column
            continue
        else:
            if np.sum(np.isnan(features[feat]))> 0.5*n_worms:
                to_exclude.append(feat)
    return to_exclude

to_excludeA={}
for rep in featuresA:
    to_excludeA[rep] = FeatFilter(featuresA[rep])

#combined for all experiments to exclude
list_exclude = [y for v in to_excludeA.values() for y in v]
list_exclude = list(np.unique(list_exclude))

#now drop these features
for rep in featuresA:
    featuresA[rep].drop(list_exclude, axis = 1, inplace= True)
    featuresA[rep] = featuresA[rep].reset_index(drop = True)

   
#%%
#from here pull out info about each of the experiment's variables, such as drug, concentration and uniqueID
import numpy as np

#make a copy of featuresA
featuresA2 = featuresA.copy()

#pop out experiment list
exp_namesA={}
for rep in featuresA:
    exp_namesA[rep] = featuresA[rep].pop('exp')

def extractVars(exp_names):
    """ Extracts out the date, drugs, concentrations and uniqueIDs from the experiments
    Input - exp_name - experiment names
    
    Output:
        date - date of recording
        
        drugs- list of drugs tested for each experiments
        
        concs - list of concentrations tested
        
        uniqueID - unique ID for plate
        """
    drug = []
    conc = []
    date = []
    uniqueID =[]
    for line in exp_names: #split the line and find the drug, concentration and date
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
    
    return drug, conc, date, uniqueID

drugA = {}
concA = {}
dateA = {}
uniqueIDA = {}
for rep in exp_namesA:
    drugA[rep], concA[rep], dateA[rep], uniqueIDA[rep] = extractVars(exp_namesA[rep])

#make lists of unqiue drugs and concs
drugs= list(drugA.values())
flatten = lambda LoL: [item for sublist in LoL for item in sublist] #lambda function flattens list of lists (LoL)

#make a list of the unqiue Drugs and concentrations
uniqueDrugs = list(np.unique(flatten(drugs)))
uniqueConcs = list(np.unique(flatten(list(concA.values()))))
uniqueDates = list(np.unique(flatten(list(dateA.values()))))

del drugs


#%% Z-score normalisation
from scipy import stats
import numpy as np

#before normalising need to average from each plate
    #add on drug, concentration, and date columns to featuresA dataframe
        # then use these to filter and average for each plate    

#add on track details
for rep in featuresA:
    featuresA[rep]['drug'] = drugA[rep]
    featuresA[rep]['concentration'] =  concA[rep]
    featuresA[rep]['date'] = dateA[rep]

#index,average and build dataframe
featMatMean = {}
for rep in featuresA:
    featMatMean[rep] = pd.DataFrame()
    for drug in uniqueDrugs:
        feats_temp = featuresA[rep][featuresA[rep]['drug']==drug]
        for conc in uniqueConcs:
            feats_temp2 = feats_temp[feats_temp['concentration'] == conc]
            if feats_temp2.empty:
                continue
            else:
                to_append = feats_temp2.mean(axis=0).to_frame().transpose()
                to_append['drug'] = drug
                to_append['concentration'] = conc
                featMatMean [rep]= featMatMean[rep].append(to_append)
    featMatMean[rep] = featMatMean[rep].reset_index(drop=True)

featuresZ = {}
drugZ = {}
concZ = {}
for rep in featMatMean:
    featuresZ[rep] = pd.DataFrame(stats.zscore(featMatMean[rep].iloc[:,4:-2]))
    featuresZ[rep].columns = list(featMatMean[rep].columns)[4:-2]
    drugZ [rep] = featMatMean[rep]['drug']
    concZ [rep] = featMatMean[rep]['concentration']
    
#drug and concentration not added to featuresZ as this interferes with future code, but use featMatMean \
    #or drugZ and concZ as lookup

#%% move on to PCA

#now to make covariance matrix for PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA

#algorithm doesn't work with NaNs, so need to impute:
    #one inputation with mean, and another with median
featMatTotalNorm_mean = {}
featMatTotalNorm_med = {}
featMatTotal_mean = {}
for rep in featuresZ:
    featMatTotalNorm_mean[rep] = featuresZ[rep].fillna(featuresZ[rep].mean(axis = 0), inplace=True)
    featMatTotalNorm_med[rep] = featuresZ[rep].fillna(featuresZ[rep].median(axis = 0), inplace = True)
    featMatTotal_mean [rep]= featuresA[rep].fillna(featuresA[rep].mean(axis=0))
    
#fit and transform data onto standard scale - this means that Z-score normalising was redundant
X_std1={}
#X_std2={}
exp_names = {}
cov_mat={}
for rep in featMatTotalNorm_mean:
    X_std1[rep] = StandardScaler().fit_transform(featMatTotalNorm_mean[rep].iloc[:,0:-2])

    cov_mat[rep] = np.cov(X_std1[rep].T)

#function defines the pca - can actually put this earlier in the script
def pca(X_std, rep, directory, file_type):
    """pca function that returns PCA scree plots and ...
    Input:
        X_std - standardly scaled raw features data
        
        rep - the name of the experiment (as in the replicate)
        
        directory - the directory for saving files
        
        file_type - type of file to save the screen plots
        
    Output:
        eig_vecs - eigen vectors (ie planes) for each of the principle components (type = ?)
        
        eig_vals - eigen values are the scaling factors for each eigenvector (type = ). Used to calculate the amount of variance explained
        
        eig_pairs - tuple containg the PC eigenvalue and array of eigenvectors for that PC - the contribution of each features tot that plane
        
        PC_pairs - tuple containing PC number, variance explained, and cumulative variance explained
        
        PC_df - dataframe of PC_pairs
        
        cut_off - integer of the number of PCs that explain 95% of the cumulative variance
        
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
    PC_df1[rep], cut_off1[rep] = pca(X_std1[rep], rep, directoryA, '.tif')
    
#now to find the top features that contribute to PC1 and PC2
def PC_feats(eig_pairs, cut_offs, features):
    """ finds the top features and returns dataframes with contributions and 
    features
    
    Input:
        eig_pairs - eigenvalue-vector tuple
        
        cut_offs - the number of PCs that contribute 95% of the variance
        
        features - features dataframe containing all the feature names - \
        use dataframe that does not include drug and conccentration
        
    Output:
        PC_contribs - list of arrays of contribution of each feature for each PC in range of cut_offs
        
        PC_features - Dataframe of PC_contribs with feature names added
        
        PC_tops - Rank list of top features contributing to each PC
        
        x - list of names of PCs
        
        """
    x = ['PC_%s' %i for i in range(1,cut_offs+1)]
    PC_contribs = [(eig_pairs[i][1]) for i in range (0,cut_offs)]
    features_1 = list(features.columns)
    PC_features = pd.DataFrame(PC_contribs)
    PC_features = PC_features.T
    PC_features.columns = x
    if len (features_1) > PC_features.shape[0]:
        features_1 = features_1[0:-2]
    else:
        features_1 = features_1
    PC_features['features'] = features_1
    
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
    PC_top1[rep], x1[rep] = PC_feats(eig_pairs1[rep], cut_off1[rep], featuresZ[rep])
    
#biplot function
def biplot(ranks, coeff, pc1, pc2, n_feats, directory, rep, file_type):
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
    biplot(PC_top1[rep], PC_feats1[rep],1,2, 1, directoryA, rep, '.tif')
    
#%% now to transform into feature space
    #concanenate the eigen_vector matrix across the top 80 eigenvalues

def feature_space(features, eig_pairs, X_std, cut_offs, x, drug, conc):
    """ transforms features data into the PC space
    Input:
        features - features dataframe after filtering
        
        eig_pairs - eig value - vector tuples
        
        X_std - standard scaled data
        
        cut_offs - number of PCs that explain 95% variance
        
        x - array of PCnames
        
        drug - list containing corresponding drugs for each row of features dataframe
        
        conc - list containing corresponding concentrtaion for each row of features dataframe
        
        date - list of dates for corresponding row of dataframe
        
    Output:
        matrix_w - matrix of features transformed into PC space
        
        Y - 
        
        PC_df - dataframe containing all the PCs for each condition
        
        """
    matrix_w = eig_pairs[0][1].reshape(eig_pairs[0][1].size,1)
    for i in range(1,cut_offs):
        temp_matrix = eig_pairs[i][1].reshape(eig_pairs[i][1].size,1)
        matrix_w = np.hstack((matrix_w, temp_matrix))
        del temp_matrix
    print ('Matrix W: \n', matrix_w)
    
    Y = X_std.dot(matrix_w)
    PC_df = pd.DataFrame(Y)
    PC_df.columns = x
    PC_df['drug'] = drug
    PC_df['concentration'] = conc
    return matrix_w, Y, PC_df

matrix_w1 = {}
Y1 = {}
PC_df2 = {}
for rep in featuresZ:
    matrix_w1[rep], Y1[rep], PC_df2[rep] = feature_space(featuresZ[rep], eig_pairs1[rep],\
            X_std1[rep], cut_off1[rep], x1[rep], drugZ[rep], concZ[rep])

#to make plots    
def PC12_plots (df, dose, rep, directory, file_type):
    """this makes plots that are scaled PCs
    Input:
        df - dataframe containing PCs for each condition
        
        dose - dose to be plotted
        
        rep - experiment name
        
        directory - directory into which the plot will be saved
        
        file_type - tif or svg
    
    Output:
        plots of each of the conditions along PCs 1 and 2
    """
    sns.set_style('whitegrid')
    sns.palplot(sns.choose_colorbrewer_palette(data_type = 'q'))
    if dose == []:
        temp = df
    else:
        to_plot = (df['concentration'] == float(dose))# or (df['concentration'] == float(14))
        temp = df[to_plot]
        temp = temp.append(df[df['drug']=='DMSO']) #add on DMSO controls
        temp = temp.append (df[df['drug'] == 'No_compound'])
    xs = temp['PC_1']
    ys = temp['PC_2']
    scalex = 1/(xs.max() - xs.min())
    scaley = 1/(ys.max() - ys.min())
    temp ['PC_1'] = temp['PC_1'].replace(temp['PC_1'].values, xs*scalex)
    temp['PC_2'] = temp['PC_2'].replace(temp['PC_2'].values, ys*scaley)
    f = plt.figure
    f= sns.lmplot(x= 'PC_1', y='PC_2', data= temp, hue = 'drug',fit_reg = False)
   
    plt.xlim (-1, 1)
    plt.ylim (-1,1)
    plt.title ('concentration = ' + str(dose))
    plt.savefig (os.path.join(directory[0:-7], 'Figures', rep + '_'\
                              + str(dose) + '_PC12_norm.' + file_type),\
                                bbox_inches="tight", dpi = 200)

#now make the plots   
for rep in PC_df2:
    for i in [1,10,100,200]:
        PC12_plots(PC_df2[rep], i, rep, directoryA, 'tif')

#now can make dataframe containing means and column names to plot trajectories through PC space
def PC_av(PC_dataframe, x):
    """function to convert to average PC for replicates. Requires PC dataframe
    and x containing all the column name
    Input:
        PC_dataframe - average value for each condition
        
        x - name of PCs
        
    Output:
        
    
    """
    PC_means= pd.DataFrame(data = None, columns = x)
    uniqueDrugs1 = np.unique(PC_dataframe['drug'])
    for drug in uniqueDrugs1:
        finders = PC_dataframe['drug'] == drug
        keepers = PC_dataframe[finders]
        concs = np.unique(keepers['concentration'])
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
    

def PC_traj(df,rep, directory, file_type):
    """this function groups by drug an plots the trajectories through PC space
    Input
        df - dataframe containing the PC values for each of the drugs
        rep - the name of the experiments
        directory - the directory to save the files into
        file_type - type of image ('tif' or 'svg' ...)
        
    Output
        Plot showing trajectory through PC space
    """ 
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
    plt.axis('scaled')
    plt.xlim (-1,1)
    plt.ylim(-1,1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5) ,ncol = 1, frameon= True)
    plt.xlabel ('PC_1')
    plt.ylabel('PC_2')
    plt.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_PCtraj.' + file_type),\
                bbox_inches="tight", dpi = 200)
    plt.show()

sns.set_style('whitegrid')
for rep in PC_means1:
    PC_traj(PC_means1[rep], rep,directoryA, 'tif')

#%% now to do the stats on the experiments
    
from scipy import stats
import seaborn as sns

#compare each compound to control data
controlMeans = {}
for rep in featuresA:
    controlMeans[rep] = pd.DataFrame()
    for line in range(len(featuresA[rep])):
        if featuresA[rep]['drug'].iloc[line] == 'DMSO':
            DMSO = featuresA[rep].iloc[line]
            controlMeans[rep] = controlMeans[rep].append(DMSO.to_frame().transpose())
            del DMSO
    controlMeans[rep].reset_index(drop=True)

#so can now compile a matrix of p_values comparing each drug to control means
    #need to loop through the different drugs and basically just compare all the features to controlMeans
pVals = {}
feats =list(featuresA[rep].columns)
feats = feats
for rep in featuresA:
    pVals[rep] = pd.DataFrame()
    for drug in uniqueDrugs:
        if drug == 'DMSO':
            continue
        else:        
            currentInds = pd.DataFrame()
            for line in featuresA[rep].iterrows():
                if line[1]['drug'] ==drug:
                    current = line[1].to_frame().transpose()
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
                    for feature in currentInds.columns[0:-3]:
                        test.append(stats.ttest_ind(testing[feature], controlMeans[rep][feature]))
       
                    ps = [(test[i][1]) for i in range(len(test))] #make into a list
                    ps.append(drug)
                    ps.append(dose)
        
                    temp = pd.DataFrame(ps).transpose()
                    pVals[rep] = pVals[rep].append(temp)
                    del temp, to_test, testing
            del currentInds

    #add in features
    pVals[rep].columns = feats[:-1]
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
    bh_p[rep].columns = feats[:-1]
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

cmap = sns.choose_colorbrewer_palette(data_type = 'q')
#make some violin plots of the significant features
def swarms (rep1, feature, features_df, directory, file_type):
    """Makes swarm plot of features
    Input:
        rep1 - name of experiment
        
        feature - feature to be plotted
        
        features_df - dataframe of features
        
        directory - folder into which figure wil be saved
        
        file_type - image type to save (eg .tif, .svg)
    
    Output:
        swarm plot - 
    """
    sns.set_style('whitegrid')
    sns.swarmplot(x='drug', y=feature, data=features_df, \
                      hue = 'concentration', palette = cmap)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5) ,ncol = 1, frameon= True)
    plt.xticks(rotation = 45)
    plt.savefig(os.path.join (directoryA[0:-7], 'Figures', rep1 + feature + file_type),\
                 bbox_inches="tight", dpi = 200) 

for rep in featuresA:
    for feat in range(0,10):
        swarms (rep, sig_feats[rep][feat][0], featuresA[rep], directoryA, '.tif')

swarms (rep, 'head_bend_mean_pos', featuresA[rep], directoryA, '.tif')

#raclopride is supposed to modulate high angle turns
drug_ps={}
for rep in pVals:
    drug_ps[rep] = {}
    for drug in uniqueDrugs:
        drug_ps[rep][drug] = bh_p[rep][pVals[rep]['drug'] == drug]
        #drug_ps[rep] [drug]= drug_ps[rep][drug].iloc[:,:-3][drug_ps[rep][drug].values[:,:-3]<=0.05]
        
        
#%% now can combine and do PCA on entire space
    #use averages for each drug and each dose - from the technical replicates
        #so first need to take averages from FeatMatNorm
            #and then concatenate to combine
                #then do PCA

featMatMean2 = pd.DataFrame()
for rep in featMatMean:
    for drug in uniqueDrugs:
        test = featMatMean[rep]['drug'] == drug
        drug_all = featMatMean[rep][test]
        for conc in uniqueConcs:
            test2 = drug_all['concentration']==conc
            if np.sum(test2) == 0:
                continue
            else:
                drug_concs = drug_all[test2]
                mean1 = drug_concs.mean(axis=0).to_frame().transpose()
                mean1['drug'] = drug
                mean1['concentration'] = conc
                mean1['experiment'] = rep
                featMatMean2 = featMatMean2.append(mean1)
                del mean1, test2, drug_concs
        del test, drug_all            
featMatMean2 = featMatMean2.reset_index(drop=True)        

#now I have the space for doing PCA
featMatMean3 = featMatMean2.copy()
all_drugs = featMatMean2.pop('drug')
all_concs = featMatMean2.pop('concentration')
all_exps = featMatMean2.pop('experiment')
featMatMean2 = featMatMean2.iloc[:,4:-1]

#standard scalar before doing the the PCA
X_std_combi  = StandardScaler().fit_transform(featMatMean2)

#PCA function
eig_vecs_combi, eig_vals_combi, eig_pairs_combi,\
PC_pairs_combi, PC_df_combi, cut_off_combi = pca(X_std_combi, 'agar', directoryA, 'tif')

#eig_pairs and biplots
PC_contribs_combi, PC_feats_combi, PC_top_combi, x_combi = \
PC_feats(eig_pairs_combi, cut_off_combi, featMatMean2)

#make the biplots
biplot(PC_top_combi, PC_feats_combi,1,2, 1, directoryA,'agar', 'tif')

#transform into feature space   
matrix_w_combi,  Y_combi, PC_df_combi2 = feature_space(featMatMean2,\
              eig_pairs_combi, X_std_combi, cut_off_combi, x_combi, all_drugs, all_concs)


#now to make plots
for i in [1,10,100,200]:
    PC12_plots(PC_df_combi2, i, 'agar', directoryA, 'tif')

#average
PC_means_combi = PC_av(PC_df_combi2, x_combi)

#plot the trajectories
PC_traj(PC_means_combi, 'agar', directoryA, 'tif')

#%% now try t-SNE on the datasset
    #this code is adapted from sci-kit learn page and analyticsvidhya.com

from time import time #for timing how long code takes to rum
from sklearn import (manifold, decomposition, ensemble)

#use the full dataset (ie including all tracks for all doses and drugs)
    #internally z-score first and then concatenate, and then run t-sne on entire dataset
    
    #test different perplexities, but keep the n_iter at 1000

#first z-score the features
featuresZ2 = {}
featuresA3 = featuresA2.copy()
featMatALLZ1 = pd.DataFrame()
for rep in featuresA3:
    #reset index
    featuresA3[rep] = featuresA3[rep].reset_index(drop=True)
    #impute nans
    featuresA3[rep] = featuresA3[rep].fillna(featuresA3[rep].mean(axis=0), inplace=True)
    #zscore
    featuresZ2[rep] = pd.DataFrame(stats.zscore(featuresA3[rep].iloc[:,4:-3], axis=0))
    featuresZ2[rep].columns = featuresA3[rep].iloc[:,4:-3].columns
    #add in drugs, doses and dates
    featuresZ2[rep]= pd.concat([featuresZ2[rep], featuresA3[rep].iloc[:,-3:]],  axis = 1)
    
    #concat into a big features dataframe
    featMatALLZ1 = featMatALLZ1.append(featuresZ2[rep])

#reset index
featMatALLZ1 = featMatALLZ1.reset_index(drop = True)

#now to do the tSNE on each dataset separately, and then the entire dataset
def tSNE_custom(features_df, to_test):
    """ Custome tSNE function for computing and iterating over tSNE perplexities 
    use sckit-learn toolbox
    Input:
        features_df - dataframe containing z-score standardised data, with NaNs
        already imputed in
        
        to_test - the perplexities to iterate over and test
        
    Output:
        X_tsne_df - dataframe containing the t-sne values for each track
        
        t0 - the time to do each tSNE computation
        """
    
    X = np.array(features_df.iloc[:,:-3]) #all values except the descriptors of the data
    #y = np.array(features_df.index) #indices for the data
    n_samples, n_features = X.shape
    
    #now to compute t-SNE
    X_tsne = {}
    X_tsne_df = {}
    t0 = {}
    for i in to_test:
        t0 [i] = time()
        print ('Computing t-SNE')
        tsne = manifold.TSNE(n_components = 2, perplexity = i, \
                             init = 'pca', random_state = 0)
        X_tsne [i] = tsne.fit_transform(X)
        t0 [i] = time() - t0[i]
        
        print (str(i) + ': ' + str(t0[i]) + 'secs')
        
        #convert to dataframe for easy plotting
        X_tsne_df[i] = pd.DataFrame (X_tsne[i])
        X_tsne_df[i].columns = ['SNE_1', 'SNE_2']
        X_tsne_df[i] = pd.concat([X_tsne_df[i], features_df.iloc[:,-3:]], axis = 1)
    
    return X_tsne_df, t0

tSNE_1 = {}
times = {}
testing = list(np.arange (1,102,20))
for rep in featuresZ2:
    tSNE_1[rep], times[rep] = tSNE_custom(featuresZ2[rep], testing)

tSNE_all, times_all = tSNE_custom(featMatALLZ1, testing)

#now to plot

def pre_plot(plotting):
    """ this plot actually juse makes tSNE scatter plots
    Input:
        plotting - dataframe containing the SNE values to plot and the drugs
    Output:
        tSNE scatter plot
    """
    sns.lmplot(x = 'SNE_1', y= 'SNE_2', data= plotting, hue = 'drug', fit_reg = False, legend = False)
    plt.axis('equal')
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.show()

def sne_plot(tSNE_df, to_plot, conc):
    """Function to plot tSNE
    Input:
        t_SNE_df - dataframe containing the two SNE dimensions and conditions - doses, drugs and dates
        to_plot - the perplexities to plot - number or array
        conc - concentration to plot
    Output:
        """
    if isinstance(to_plot, list):
        if conc == []:
            for i in to_plot:
                plotting = tSNE_df[i][tSNE_df[i]['concentration'].isin(uniqueConcs)]
                pre_plot(plotting)
        else:
            for i in to_plot:
                plotting = tSNE_df[i][tSNE_df[i]['concentration']==float(conc)]
                plotting = plotting.append(tSNE_df[i][tSNE_df[i]['drug']=='DMSO'])
                plotting = plotting.append(tSNE_df[i][tSNE_df[i]['drug'] == 'No_compound'])
                pre_plot(plotting)
                del plotting
    else:
        i = to_plot
        if conc == []:
            plotting = tSNE_df[i][tSNE_df[i]['concentration'].isin(uniqueConcs)]
            pre_plot(plotting)
        else:
            plotting = tSNE_df[i][tSNE_df[i]['concentration']==float(conc)]
            plotting = plotting.append(tSNE_df[i][tSNE_df[i]['drug']=='DMSO'])
            plotting = plotting.append(tSNE_df[i][tSNE_df[i]['drug'] == 'No_compound'])
            pre_plot(plotting)
    
sne_plot(tSNE_1[rep], testing, 100)

sne_plot(tSNE_all, testing,100)

#okay so what about using featMatTotalNorm_mean
featMatTotalAll = pd.DataFrame()
for rep in featMatTotalNorm_mean:
    temp = featMatTotalNorm_mean[rep]
    temp = pd.concat([temp, drugZ[rep], concZ[rep]], axis=1)
    temp['exp'] = rep
    featMatTotalAll = featMatTotalAll.append(temp)
#reset index
featMatTotalAll = featMatTotalAll.reset_index(drop = True)

#do tSNE on this data
tSNE_all2, t0_2 = tSNE_custom(featMatTotalAll, testing)
sne_plot(tSNE_all2, testing[2], [])    

#%%clustering
#first attempt to replicate clusterdv from Marqus et al 2017;2018. 

#first step is to fit kernel disributions to the data
    #on the tSNE data

#need to find optimum bandwidth
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

#may be best to fit grid to max and min of tSNE dimensions
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.1, 1.0, 30)},
                    cv=20) # 20-fold cross-validation

grid.fit(tSNE_all[21].iloc[:,0:2], None)
print (grid.best_params_)

kde = grid.best_estimator_
pdf = np.exp(kde.score_samples(tSNE_all[21].iloc[:,0:2]))

fig, ax = plt.subplots()
ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
ax.legend(loc='upper left')
ax.set_xlim(-4.5, 3.5);