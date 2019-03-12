#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:59:57 2017

@author: ibarlow
"""
"""script for multiworm tracker to analyse data from worms treated
#antipsychotics - done by Adam; 3 concentrations for each drug; 10 worms; 3 replicates """

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
for repeat in reps:
    if repeat != '.DS_Store':
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

#now save these 116 features - the averages from the experiments (ie features 2)
#hdf = pd.HDFStore('psychotics_summary.h5') #create file
#for rep in features2:
 #   df = features2[rep]
  #  hdf.put (rep, df, format = 'table')

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
X_std={}
#X_std2={}
exp_names = {}
cov_mat={}
cov_mat2 ={}
for rep in features2:
    X_std[rep] = StandardScaler().fit_transform(featMatTotalNorm_mean[rep].iloc[:,4:-2])
    #X_std2[rep] = StandardScaler().fit_transform(features2[rep].iloc[:,4:-2]) #don't include the recording info in the PCA

    exp_names [rep] = features2[rep]['experiment']

    cov_mat[rep] = np.cov(X_std[rep].T)
    #cov_mat2[rep] = np.cov(X_std2[rep].T)

#old code ----
#eigen decomposition of covariance matrix
#eig_vals[rep], eig_vecs[rep] = np.linalg.eig(cov_mat)

#nb the above can be done more efficiently by performing singular value decomposition
eig_vecs2 = {}
eig_vals2 = {}
for rep in X_std:
    u,s,v = np.linalg.svd(X_std[rep].T)
    eig_vecs2[rep] = u
    eig_vals2[rep] = s
    del u,s,v
# u and eig_vecs, and s and eig_vals are the same - eig_vals gave complex number so use u and s for further analysis
#eig_vecs2 = u
#eig_vals2 = s

#unit of eigen vectors should be 1
for rep in eig_vecs2:
    for ev in eig_vecs2[rep]:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print ("Everything OK!")

#now need to sort the eigenvectors
    #make a list of the eigenvalue-vector as a tuple
eig_pairs = {}
for rep in eig_vecs2:
    eig_pairs[rep] = [(np.abs(eig_vals2[rep][i]), eig_vecs2[rep][:,i]) for i in range(len(eig_vals2[rep]))]

    #then sort - high to low
    eig_pairs[rep].sort(key =lambda tup:tup[0])
    eig_pairs[rep].reverse()

    print('Eigenvalues in descending order:')
    for i in eig_pairs[rep]:
        print(i[0])
    
#how many principle components explain the variance - do this for each experiment separately
PC_pairs = {}
PC_df = {}
cut_off={}
sns.palplot(sns.color_palette("husl", 8))

#make scree-like plots
for rep in eig_vals2:
    tot = sum(eig_vals2[rep])
    var_exp = [(i / tot)*100 for i in sorted(eig_vals2[rep], reverse=True)]
    cum_var_exp= np.cumsum(var_exp)
    #add in cut off for where 95% variance explained
    cut_off [rep]= cum_var_exp <95
    cut_off [rep]= np.argwhere(cut_off[rep])
    cut_off [rep] = int(cut_off[rep][-1])

    #first make dataframe with all the PCs in
    x=['PC %s' %i for i in range(1,len(eig_vals2[rep]))]
    y= var_exp[0:len(eig_vals2[rep])]
    z=cum_var_exp[0:len(eig_vals2[rep])]
    PC_pairs [rep] = [(x[i], y[i], z[i]) for i in range(0,len(eig_vals2[rep])-1)]
    PC_df[rep] = pd.DataFrame(data=PC_pairs[rep], columns = ['PC', 'variance_explained', \
                                               'cum_variance_explained'])
    f, (ax1,ax2) = plt.subplots(ncols=2, sharey=True)
    plt.title(rep)
    trace1 = sns.barplot(y= 'variance_explained', x= 'PC', data=PC_df[rep], ax=ax1)
    sns.despine()
    ax1.xaxis.set_ticks(np.arange(0,50,10))
    ax1.xaxis.set_ticklabels(PC_df[rep]['PC'][0:51:10])
    trace2 = sns.stripplot(y='cum_variance_explained', x='PC', data=PC_df[rep], ax=ax2)
    ax2.xaxis.set_ticks(np.arange(0,50,10))
    ax2.xaxis.set_ticklabels(PC_df[rep]['PC'][0:51:10])
    trace2 = plt.plot([cut_off[rep], cut_off[rep]], [0,95], linewidth =2)
    plt.text(cut_off[rep], 100, str(cut_off[rep]))
    
    sns.despine()
    f.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_PC_variance_explained.tif'), dpi=400)
    f.show()
    del x,y, z, tot, var_exp, cum_var_exp, f, ax1, ax2, trace1, trace2

#basically need 35 eigenvectors to explain 95% of the variance - this was for only one experiment. 

#%% now analyse how the features contribute to the eigenplanes
#see how the features contribute to the first 10 eigenvectors

#for each rep - do combined later
PC_feats = {}
#set style of plot
sns.set_style ('whitegrid')
sns.set_style("ticks")
cmap = sns.color_palette("husl", len(uniqueDrugs))

for rep in eig_pairs:
    x = ['PC_%s' %i for i in range (1,cut_off[rep]+1)]
    PC_conts = [(eig_pairs[rep][i][1]) for i in range (0,cut_off[rep])]
    feats = list(features2[rep].columns[4::1]) #make array with feature names
    PC_feats[rep] = pd.DataFrame(PC_conts)
    PC_feats[rep] = PC_feats[rep].T
    PC_feats[rep].columns = x
    PC_feats[rep]['features'] = feats[0:-3] #concentration and experiment not included
    del x, PC_conts
    
    #make a plot of the first 4
    f, (subplot) = plt.subplots(nrows=4, ncols =1, sharey=True, sharex=True)
    for current in range(0,4):
        sns.pointplot(x='features', y=PC_feats[rep].columns[current], data=PC_feats[rep], \
                      markers = '.', color =cmap[current], linewidth = 0.5, ax= subplot[current])
        [tick.set_rotation(45) for tick in subplot[current].get_xticklabels()]
        f.set_size_inches(20, 20)
        f.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_PCS.tif'), dpi = 400)

#update 8th Jan - simply rank the top features for all the PCs, and then these can 
#be plotted on the biplots
#make a dictionary of name value pairs
PC_top ={}
for rep in PC_feats:
    PC_top[rep] = {}
    for PC in PC_feats[rep].columns:
        PC_top[rep][PC] = list(PC_feats[rep][PC].sort_values().index[:])
        PC_top[rep][PC].reverse()
        PC_top[rep][PC] = PC_feats[rep]['features'].iloc[PC_top[rep][PC]]

#biplot function
def biplot(ranks, coeff, pc1, pc2, n_feats, rep):
    """ biplot function """
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
                    plt.text (coeff[np.flip(pcs,axis =0)[pc]].iloc[ranks[pcs[pc]].index[i]]*1.5, \
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
                              coeff[np.flip(pcs,axis =0)[pc]].iloc[ranks[pcs[pc]].index[i]]*1.5,\
                              coeff['features'].iloc[ranks[pcs[pc]].index[i]], color = cmap[pc],\
                              ha = 'center', va='center')
    plt.xlim (-1, 1)
    plt.ylim (-1,1)
    #plt.axis('equal')
    plt.xlabel ('PC_1')
    plt.ylabel('PC_2')
    plt.legend()
    plt.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_biplot.tif'), dpi =200)
    plt.show()

#now make biplots for all the reps 
for rep in PC_top:
    biplot(PC_top[rep], PC_feats[rep],1,2, 1, rep)
    
#%% now to make the feature space
    #concanenate the eigen_vector matrix across the top 50 eigenvalues

#matrix of eigenvectors - the top PCs for each condition
matrix_w = {}
Y={}
Y_sorted={}
for rep in eig_pairs:
    matrix_w[rep] = eig_pairs[rep][0][1].reshape(eig_pairs[rep][0][1].size,1)
    for i in range(1,cut_off[rep]):
        temp_matrix = eig_pairs[rep][i][1].reshape(eig_pairs[rep][i][1].size,1) 
        matrix_w [rep]= np.hstack((matrix_w[rep], temp_matrix))
        del temp_matrix
    print('Matrix W:\n', matrix_w[rep])

    #now transform onto new feature subspace - multiply standard scalar by PC
    Y [rep]= X_std[rep].dot(matrix_w[rep])

    Y_sorted[rep] = {}
    for drug in uniqueDrugs:
        Y_sorted[rep][drug] = []
        for line in range(0,len(exp_names[rep])):
            match = re.search(drug,exp_names[rep][line])
            if match:
                conc = float(re.search(r'\d+', \
                                   exp_names[rep][line][match.end()+1:match.end()+4]).group()) #find the concentration
                tempY = Y[rep][line]
                tempY = np.append (tempY, conc) #add concentration on as the last column
                Y_sorted[rep][drug].append(tempY)
                del tempY, conc
            
del drug, match

#reshape for easy plotting
Y_sorted2 ={}
PC_list = {}
PC_df2 = {}
x={}
for rep in Y_sorted:
    Y_sorted2[rep] ={}
    for drug in Y_sorted[rep]:
        if len(Y_sorted[rep][drug]) == 0:
            continue
        else:
            Y_sorted2[rep][drug] = Y_sorted[rep][drug][0].reshape(1,Y_sorted[rep][drug][0].size)
            for line in range(1,len(Y_sorted[rep][drug])):
                temp= Y_sorted[rep][drug][line].reshape(1,Y_sorted[rep][drug][line].size)
                Y_sorted2[rep][drug] = np.vstack((Y_sorted2[rep][drug], temp))

    PC_list[rep] = [list(zip([k]*len(v),v)) for k,v in Y_sorted2[rep].items()]
    PC_list[rep] = sum(PC_list[rep],[])
    drugs_only, values = zip(*PC_list[rep])
    values = np.array(values)

    PC_df2[rep] = pd.DataFrame(values[:,0:-1])
    PC_df2[rep]['drug'] = drugs_only
    PC_df2[rep]['concentration'] = values[:,-1]
    x[rep]=['PC_%s' %i for i in range(1, cut_off[rep]+1)]
    PC_df2[rep].columns.values[0:cut_off[rep]] = x[rep]
    del values, drugs_only

#----- old code ---------
#make list of tuples for plotting the concentrations
#concs = (np.unique(PC_df2[rep]['concentration']),\
 #        1/(np.arange(len(np.unique(PC_df2['concentration']))+1,1,-1)-1))
#-----------------------#
 
#now make plots

def PC12_plots (df, dose, rep):
    """this makes plots that are scaled PCs"""
    sns.set_style('whitegrid')
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
                              + str(dose) + '_PC12_norm.tif'), dpi = 200)

#now make the plots   
for rep in PC_df2:
    PC12_plots(PC_df2[rep], 100, rep)
    PC12_plots(PC_df2[rep], 1,rep)
    PC12_plots(PC_df2[rep], 10,rep)
    PC12_plots(PC_df2[rep], 200, rep)

#and now plot trajectory of doses through PC space
    #first need to average the technical replicates for each dose
#add drug and dose onto list of PC names to make columns for dataframe
for rep in x:
    x[rep].append('drug')
    x[rep].append ('concentration')

#now can make dataframe containing means and column names
PC_means = {}
for rep in PC_df2:
    PC_means[rep]= pd.DataFrame(data = None, columns = x[rep])
    for drug in uniqueDrugs:
        finders = PC_df2[rep]['drug'] == drug
        keepers = PC_df2[rep][finders]
        concs = plt.unique(keepers['concentration'])
        for dose in concs:
            refine = keepers['concentration'] == dose
            final = keepers[refine]
            temp = final.iloc[:,0:-2].mean(axis=0)
            temp = temp.to_frame().transpose()
            temp['drug'] = drug
            temp['concentration'] = dose
            PC_means[rep] = PC_means[rep].append(temp)
            del refine, final, temp
        del finders, keepers, concs

    PC_means[rep] = PC_means[rep].reset_index(drop=True)

#now plot trajectories of each drug through the PC space

def PC_traj(df,rep):
    """this function groups by drug an plots the trajectories through PC space"""
    #scale the PCs
    xscale = 1/(np.max(PC_means[rep]['PC_1']) - np.min(PC_means[rep]['PC_1']))
    yscale = 1/(np.max(PC_means[rep]['PC_2']) - np.min(PC_means[rep]['PC_2']))
    #okay so now have a summary of each drug for each PC.
        #scale and plot the drugs across the PC1 and 2 space
    for drug in range(len(np.unique(PC_means[rep]['drug']))):
        to_plot = PC_means[rep]['drug'] == uniqueDrugs[drug]
        plotting1 = PC_means[rep][to_plot]
        ax = plt.plot(plotting1['PC_1']*xscale, plotting1['PC_2']*yscale, \
                      linewidth =2, color = cmap[drug], marker = 'o', \
                      label = uniqueDrugs[drug])
    plt.xlim (-1,1)
    plt.ylim(-1,1)
    plt.legend(loc = 2,ncol = 3, frameon= True)
    plt.xlabel ('PC_1')
    plt.ylabel('PC_2')
    plt.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_PCtraj.svg'), dpi = 200)
    plt.show()

sns.set_style('whitegrid')
for rep in PC_means:
    PC_traj(PC_means[rep], rep)
    
#or as separate facets
grid = sns.FacetGrid(PC_means, col = 'drug', hue = 'drug')
grid.map(plt.plot,'PC_1', 'PC_2')    


#old plots
#for dose in range(0,len(concs[0])):
 #   to_plot = PC_df2['concentration']==concs[0][dose]
  #  temp = PC_df2[to_plot]
   # ax = sns.lmplot(x='PC_1', y='PC_2', data=temp,\
    #        hue = 'drug', fit_reg=False, scatter_kws={'alpha': concs[1][dose]})
    #plt.xlim(-25,25)
    #plt.ylim(-25,25)
    #plt.savefig(directory + '/' + 'PCA_1_' + str(concs[0][dose]) + '.tif')
#plt.show()


#%% now to do some statistics to compare features
    #rank sum tests and t-test
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
                    for feature in currentInds.columns[4:-2]:
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
        reg, corrP, t,s =  smm.multipletests(pVals[rep].values[cond, 0:-2],\
                                             alpha=0.05, method = 'fdr_bh', \
                                             is_sorted = False, returnsorted = False)
        corrP = list(corrP)
        corrP.append(pVals[rep]['experiment'].iloc[cond])
        corrP.append(pVals[rep]['concentration'].iloc[cond])
        corrP = pd.DataFrame(corrP).transpose()
        bh_p [rep]= bh_p[rep].append(corrP)
        del corrP
        #add in the feature names
    bh_p[rep].columns = feats[0:-1]
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
for rep in sig_feats:
    for feat in range(0,10):
        sns.swarmplot(x='drug', y='worm_dwelling', data=features3[rep], \
                      hue = 'concentration', palette = cmap)
        plt.xticks (rotation= 45)
        plt.show()
        

#old code---------------
#remove features with no sig p-val from the PC_feats dataframe for creating a new biplot
    #make an index of the rows to remove
i1= []
for feat in post_exclude:
    i1.append(PC_feats.index[PC_feats['features'] == feat])
#make into a list
i2 = [(i1[e][0]) for e in range(len(i1))]

#and now can drop from features and PCs
PC_feats2 = PC_feats
PC_feats2.drop(i2,axis=0, inplace=True)       


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
                mean1['date'] = rep
                featMatNorm2 = featMatNorm2.append(mean1)
                del mean1, test2, drug_concs
        del test, drug_all            
featMatNorm2 = featMatNorm2.reset_index(drop=True)

#now I have the space for doing PCA
    #however there are nans in the dataset (due to the liquid dataset)
        #1. remove the liquids and do PCA
        #2. keep liquids and interpolate the means

featMatNorm3 = pd.DataFrame(featMatNorm2[featMatNorm2['date']!=reps[3]])
featMatNorm4 = featMatNorm2.fillna(featMatNorm2.mean(axis = 0), inplace=True)

#standard scalar before doing the the PCA
X_std_combi = {}
X_std_combi ['all'] = StandardScaler().fit_transform(featMatNorm4.iloc[:,4:-3])
X_std_combi ['agar'] = StandardScaler().fit_transform(featMatNorm3.iloc[:,4:-3])


#%% function defines the pca - can actually put this earlier in the script
def pca(X_std, rep):
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
    f, (ax1,ax2) = plt.subplots(ncols=2, sharey=True)
    plt.title(rep)
    trace1 = sns.barplot(y= 'variance_explained', x= 'PC', data=PC_df, ax=ax1)
    sns.despine()
    ax1.xaxis.set_ticks(np.arange(0,50,10))
    ax1.xaxis.set_ticklabels(PC_df['PC'][0:51:10])
    trace2 = sns.stripplot(y='cum_variance_explained', x='PC', data=PC_df, ax=ax2)
    ax2.xaxis.set_ticks(np.arange(0,50,10))
    ax2.xaxis.set_ticklabels(PC_df['PC'][0:51:10])
    trace2 = plt.plot([cut_off, cut_off], [0,95], linewidth =2)
    plt.text(cut_off, 100, str(cut_off))
    
    sns.despine()
    f.savefig(os.path.join(directory[0:-7], 'Figures', rep + '_PC_variance_explained.tif'), dpi=400)
    f.show()
    del x,y, z, tot, var_exp, cum_var_exp, f, ax1, ax2, trace1, trace2
    
    return eig_vecs, eig_vals, eig_pairs, PC_pairs, PC_df, cut_off

eig_vecs_combi = {}
eig_vals_combi = {}
eig_pairs_combi = {}
PC_pairs_combi = {}
PC_df_combi={}
cut_off_combi={}
for cond in X_std_combi:
    eig_vecs_combi[cond], eig_vals_combi[cond], eig_pairs_combi[cond],\
    PC_pairs_combi[cond], PC_df_combi[cond], cut_off_combi[cond] = pca(X_std_combi[cond], str(cond))


#now to find the top features that contribute to PC1 and PC2

def PC_feats2(eig_pairs3, cut_offs, features_2):
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

PC_conts_combi = {}
PC_feats_combi = {}
PC_top_combi={}
x_combi ={}
for cond in eig_pairs_combi:
    PC_conts_combi[cond], PC_feats_combi[cond], \
    PC_top_combi[cond], x_combi[cond] = PC_feats2(eig_pairs_combi[cond], cut_off_combi[cond], featMatNorm2)

#make the biplots
for cond in PC_top_combi:
    biplot(PC_top_combi[cond], PC_feats_combi[cond],1,2, 1, cond)


#now can make the final PCA plots
    #the feature space
    #concanenate the eigen_vector matrix across the top 50 eigenvalues


def feature_space(features_1, eig_pairs1, X_std1, cut_offs, x1):
    matrix_w1 = eig_pairs1[0][1].reshape(eig_pairs1[0][1].size,1)
    for i in range(1,cut_offs):
        temp_matrix1 = eig_pairs1[i][1].reshape(eig_pairs[i][1].size,1)
        matrix_w1 = np.hstack((matrix_w1, temp_matrix1))
        del temp_matrix1
    print ('Matrix W: \n', matrix_w1)
    
    Y1 = X_std1.dot(matrix_w1)
    PC_df3 = pd.DataFrame(Y1)
    PC_df3.columns = x1
    PC_df3['drug'] = features_1['drug']
    PC_df3['concentration'] = features_1['concentration']
    PC_df3['date'] = features_1['date']
    return matrix_w1, Y1, PC_df3

matrix_w_combi={}
Y_combi = {}
PC_df_combi ={}
    
matrix_w_combi['all'],  Y_combi['all'], PC_df_combi['all'] = feature_space(featMatNorm4,\
              eig_pairs_combi['all'], X_std_combi['all'], cut_off_combi['all'], \
              x_combi['all'])

matrix_w_combi['agar'],  Y_combi['agar'], PC_df_combi['agar'] = feature_space(featMatNorm3,\
              eig_pairs_combi['agar'], X_std_combi['agar'], cut_off_combi['agar'], \
              x_combi['agar'])
    
#now to make plots
 
for cond in PC_df_combi:
    PC12_plots(PC_df_combi[cond], 1, cond)
    PC12_plots(PC_df_combi[cond], 10, cond)
    PC12_plots(PC_df_combi[cond], 100, cond)
    PC12_plots(PC_df_combi[cond], 200, cond)

#now can make dataframe containing means and column names to plot trajectories through PC space
def PC_av(PC_dataframe, x1):
    """function to convert to average PC for replicates. Requires PC dataframe
    and x containing all the column name"""
    PC_means1= pd.DataFrame(data = None, columns = x1)
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
            PC_means1= PC_means1.append(temp)
            del refine, final, temp
        del finders, keepers, concs

    PC_means1 = PC_means1.reset_index(drop=True)
    return PC_means1

PC_means_combi={}
for cond in PC_df_combi:
    PC_means_combi[cond] = PC_av(PC_df_combi[cond], x_combi[cond])
   
for cond in PC_means_combi:
    PC_traj(PC_means_combi[cond], cond)