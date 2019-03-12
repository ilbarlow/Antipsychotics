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
import TierPsyInput as TP


#now call these functions
directoryA, fileDirA, featuresA, trajectoriesA = TP.TierPsyInput('old', 'Liquid')

#%%
#now for filtering of tracks and features

import numpy as np

to_excludeA={}
for rep in featuresA:
    if featuresA[rep].empty:
        continue
    else:
        to_excludeA[rep] = TP.FeatFilter(featuresA[rep])

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
    if featuresA[rep].empty:
        continue
    else:
        exp_namesA[rep] = featuresA[rep]['exp']

drugA = {}
concA = {}
dateA = {}
uniqueIDA = {}
for rep in exp_namesA:
    drugA[rep], concA[rep], dateA[rep], uniqueIDA[rep] = TP.extractVars(exp_namesA[rep])

#make lists of unqiue drugs and concs
drugs = []
concs = []
for rep in drugA:
    drugs.append(list(drugA[rep].values))
    concs.append(list(concA[rep].values))
flatten = lambda LoL: [item for sublist in LoL for item in sublist] #lambda function flattens list of lists (LoL)

uniqueDrugs = list(np.unique(flatten(drugs)))
uniqueConcs = list(np.unique(flatten(concs)))
del drugs, concs


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
    featuresZ[rep] = pd.DataFrame(stats.zscore(featMatMean[rep].iloc[:,4:-3]))
    featuresZ[rep].columns = list(featMatMean[rep].columns)[4:-3]
    drugZ [rep] = featMatMean[rep]['drug']
    concZ [rep] = featMatMean[rep]['concentration']
    
#drug and concentration not added to featuresZ as this interferes with future code, but use featMatMean \
    #or drugZ and concZ as lookup

#%% move on to PCA

#now to make covariance matrix for PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import PCA_analysis as PC
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
    X_std1[rep] = StandardScaler().fit_transform(featMatTotalNorm_mean[rep])

    cov_mat[rep] = np.cov(X_std1[rep].T)
    
#pca    
eig_vecs1={}
eig_vals1 = {}
eig_pairs1 = {}
PC_pairs1={}
PC_df1 = {}
cut_off1={}

for rep in X_std1:
    eig_vecs1[rep], eig_vals1[rep], eig_pairs1[rep], PC_pairs1[rep],\
    PC_df1[rep], cut_off1[rep] = PC.pca(X_std1[rep], rep, directoryA, '.tif')
    
#now to find the top features that contribute to PC1 and PC2
PC_conts1 = {}
PC_feats1 = {}
PC_top1={}
x1 ={}
for rep in eig_pairs1:
    PC_conts1[rep], PC_feats1[rep], \
    PC_top1[rep], x1[rep] = PC.PC_feats(eig_pairs1[rep], cut_off1[rep], featuresZ[rep])

#now make biplots for all the reps 
for rep in PC_top1:
    PC.biplot(PC_top1[rep], PC_feats1[rep],1,2, 1, directoryA, rep, '.tif', uniqueDrugs)
    
#%% now to transform into feature space
    #concanenate the eigen_vector matrix across the top 80 eigenvalues

matrix_w1 = {}
Y1 = {}
PC_df2 = {}
for rep in featuresZ:
    matrix_w1[rep], Y1[rep], PC_df2[rep] = PC.feature_space(featuresZ[rep], eig_pairs1[rep],\
            X_std1[rep], cut_off1[rep], x1[rep], drugZ[rep], concZ[rep], dateA[rep])


#now make the plots   
for rep in PC_df2:
    for i in [1,10,100,200]:
        PC.PC12_plots(PC_df2[rep], i, rep, directoryA, 'tif')

#now can make dataframe containing means and column names to plot trajectories through PC space
PC_means1={}
for rep in PC_df2:
    PC_means1[rep] =PC.PC_av(PC_df2[rep], x1[rep])
    
sns.set_style('whitegrid')
for rep in PC_means1:
    PC.PC_traj(PC_means1[rep], rep,directoryA, 'tif')

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
featMatMean2 = featMatMean2.reset_index(drop=True)

#standard scalar before doing the the PCA
X_std_combi  = StandardScaler().fit_transform(featMatMean2)

#PCA function
eig_vecs_combi, eig_vals_combi, eig_pairs_combi,\
PC_pairs_combi, PC_df_combi, cut_off_combi = PC.pca(X_std_combi, 'agar', directoryA, 'tif')

#eig_pairs and biplots
PC_contribs_combi, PC_feats_combi, PC_top_combi, x_combi = \
PC.PC_feats(eig_pairs_combi, cut_off_combi, featMatMean2)

#make the biplots
PC.biplot(PC_top_combi, PC_feats_combi,1,2, 1, directoryA,'agar', 'tif')

#transform into feature space   
matrix_w_combi,  Y_combi, PC_df_combi2 = PC.feature_space(featMatMean2,\
              eig_pairs_combi, X_std_combi, cut_off_combi, x_combi, all_drugs, all_concs, all_exps)


#now to make plots
for i in [1,10,100,200]:
    PC.PC12_plots(PC_df_combi2, i, 'agar', directoryA, 'tif')

#average
PC_means_combi = PC.PC_av(PC_df_combi2, x_combi)

#plot the trajectories
PC.PC_traj(PC_means_combi, 'agar', directoryA, 'tif')

#%% now try t-SNE on the datasset
    #this code is adapted from sci-kit learn page and analyticsvidhya.com

import tSNE_custom as SNE

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


tSNE_1 = {}
times = {}
testing = list(np.arange (1,102,20))

#know that perplexity =20 is best to just use that
to_test =20
for rep in featuresZ2:
    tSNE_1[rep], times[rep] = SNE.tSNE_custom(featuresZ2[rep], 20)

tSNE_all, times_all = SNE.tSNE_custom(featMatALLZ1, 20)

SNE.sne_plot(tSNE_1[rep],20, [], uniqueConcs)

SNE.sne_plot(tSNE_all, 20, [], uniqueConcs)

#%%

#just try using Dcluster package
from scipy import spatial
import Dcluster as dcl
import os
from kneed import KneeLocator
import density_plots as dp

temp1 = spatial.distance.pdist(tSNE_all[to_test].iloc[:,0:2])
temp1 = spatial.distance.squareform(temp1)

#this make squareform distance matrix so assume NDx == NDy

#save temp1 as a .txt file
fileid1 = os.path.join(directoryA[0:-7], 'old_feat_pdist.txt')
np.savetxt(fileid1, temp1, delimiter = '\t', fmt = '%f')

#also save the file identifiers
fileid2 = os.path.join(directoryA[0:-7], 'old_ids.csv')
tSNE_all[to_test].to_csv(fileid2)

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

y = np.sort(-rho1*delta1)
x = np.arange(len(rho1))

sns.set_style('whitegrid')
f,ax, rho_d = dp.plot1(rho1, delta1)
f.savefig(os.path.join(directoryA[0:-7], 'Figures', 'old_decision_graph.png'), dpi =200)

print(rho_d)

#determine rhomin and deltmin
rhomin1 = 50
deltamin1 = 10

ax, clusters1 = dp.density_calc (dist1, XY1, ND1, rho1, delta1,ordrho1,dc1,nneigh1,rhomin1, deltamin1, directoryA[0:-7])
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'decision_graph2.png'), dpi =200)

dc_df = pd.DataFrame (data = XY1, columns = ['x', 'y'])
dc_df['cluster'] = clusters1[:,1]
dc_df['cluster_halo'] = clusters1[:,2]
dc_df['drug'] =ids['drug']
dc_df['concentration'] = ids['concentration']
N_clusters = int(max(clusters1[:,1]))

#make the plots
cmap = sns.color_palette("tab20", N_clusters)
sns.lmplot(x= 'x', y='y', data=dc_df, hue = 'cluster', fit_reg = False, legend=False)
plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
plt.tight_layout(rect=[0,0,0.75,1])
plt.axis('equal')
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'clusters__only_old.png'), dpi = 200)

sns.lmplot(x= 'x', y='y', data=dc_df, hue = 'drug', fit_reg = False, legend=False, palette =cmap)
plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
plt.tight_layout(rect=[0,0,0.75,1])
plt.axis('equal')
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'drugs_clusters_old.png'), dpi = 200, bbox_inches = 'tight')

#%% old from here

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

#%%
#just try using Dcluster package - Rodriguez and Laio Science 2014

from scipy import spatial
import Dcluster as dcl
import os

temp1 = spatial.distance.pdist(tSNE_all[21].iloc[:,0:2])
temp1 = spatial.distance.squareform(temp1)


#now have squareform distance matrix
    #need to reshape for use in Dcluster

test = [(temp1[t2].T) for t2 in range(0,len(temp1))]
test = np.concatenate(test[:])
index1 = [(np.linspace(i,i, len(temp1[i]))) for i in range(0,len(temp1))]
index1= np.concatenate (index1[:])
index1= index1+1

#make lists for the indices
index2 = [(np.arange(0, len(temp1[i]))) for i in range(0,len(temp1))]
index2 = np.concatenate(index2[:])
index2 = index2+1

final = np.vstack((index1, index2, test)).transpose()

fileid = os.path.join(directoryA[0:-7], 'test_old_features.dat')
fileid = '/Volumes/behavgenom_archive$/Adam/screening/antipsychotics/test_old_features.dat'

np.savetxt(fileid, final, delimiter = '\t', fmt = ('%d', '%d', '%4f'))

sns.set()

#try running the function stepwise
#xx = np.genfromtxt(file, delimiter=sep,names=['x','y','dist'],dtype="i8,i8,f8")
    # ND: number of data point
X = final[:,0]
Y = final[:,1]
xxdist1 = final[:,2]
ND1 = Y.max()
NL = X.max()
if NL>ND1:
    ND = NL
    # N: number of point pairs/distance
N1 = final.shape[0]
dist1 = np.zeros((int(ND1), int(ND1)))

# dist may save half of memory
dist1 = temp1

(dist1, xxdist1, ND1 ,N1) = dcl.readfile(file = fileid, sep = '\t')
(Y1, S1) = dcl.mds(dist1)
(rho1, delta1, ordrho1,dc1, nneigh1) = dcl.rhodelta(dist1, xxdist1, int(ND1), N1)

#function lifted from package to help determine rhomin and deltamin
def plot1(rho, delta):
    f, axarr = plt.subplots(1,2)
    axarr[0].set_title('DECISION GRAPH')
    axarr[0].scatter(rho, delta, alpha=0.6,c='black')
    axarr[0].set_xlabel(r'$\rho$')
    axarr[0].set_ylabel(r'$\delta$')
    axarr[1].set_title('DECISION GRAPH 2')
    axarr[1].scatter(np.arange(len(rho))+1, -np.sort(-rho*delta), alpha=0.6,c='black')
    axarr[1].set_xlabel('Sorted Sample')
    axarr[1].set_ylabel(r'$\rho*\delta$')
    return(f,axarr)

sns.set_style('whitegrid')
plot1(rho1, delta1)

rhomin = 10
deltamin = 4

def DCplot(dist, XY, ND, rho, delta,ordrho,dc,nneigh, rhomin,deltamin):
    #f, axarr = plot1(rho, delta)
    print('Cutoff: (min_rho, min_delta): (%.2f, %.2f)' %(rhomin,deltamin))
    NCLUST = 0
    cl = np.zeros(ND)-1
    # 1000 is the max number of clusters
    icl = np.zeros(1000)
    for i in range(ND):
        if rho[i]>rhomin and delta[i]>deltamin:
            cl[i] = int(NCLUST)
            icl[NCLUST] = int(i)
            NCLUST = NCLUST+1

    print('NUMBER OF CLUSTERS: %i'%(NCLUST))
    print('Performing assignation')
    # assignation
    for i in range(ND):
        if cl[ordrho[i]]==-1:
            cl[ordrho[i]] = cl[int(nneigh[ordrho[i]])]

    #halo
    # cluster id start from 1, not 0
    ## deep copy, not just reference
    halo = np.zeros(ND)
    halo[:] = cl

    if NCLUST>1:
        bord_rho = np.zeros(NCLUST)
        for i in range(ND-1):
            for j in range((i+1),ND):
                if cl[i]!=cl[j] and dist[i,j]<=dc:
                    rho_aver = (rho[i]+rho[j])/2
                    if rho_aver>bord_rho[int(cl[i])]:
                        bord_rho[int(cl[i])] = rho_aver
                    if rho_aver>bord_rho[int(cl[j])]:
                       bord_rho[int(cl[j])] = rho_aver
            for i in range(ND):
                if rho[i]<bord_rho[int(cl[i])]:
                    halo[i] = -1

    for i in range(NCLUST):
        nc = 0
        nh = 0
        for j in range(ND):
            if cl[j]==i:
                nc = nc+1
                if halo[j]==i:
                    nh = nh+1
            print('CLUSTER: %i CENTER: %i ELEMENTS: %i CORE: %i HALO: %i'%( i+1,icl[i]+1,nc,nh,nc-nh))
        # print , start from 1
        
        ## save CLUSTER_ASSIGNATION
        print('Generated file:CLUSTER_ASSIGNATION')
        print('column 1:element id')
        print('column 2:cluster assignation without halo control')
        print('column 3:cluster assignation with halo control')
        clusters = np.array([np.arange(ND)+1,cl+1,halo+1]).T
        np.savetxt('CLUSTER_ASSIGNATION_%.2f_%.2f_.txt'%(rhomin,deltamin),clusters,fmt='%d\t%d\t%d')
        print('Result are saved in file CLUSTER_ASSIGNATION_%.2f_%.2f_.txt'%(rhomin,deltamin))
        ################# plot the data points with cluster labels
        #cmap = cm.rainbow(np.linspace(0, 1, NCLUST))
        #plot2(axarr,rho, delta,cmap,cl,icl,XY,NCLUST)
        #f.show()
        #figure = plt.gcf() # get current figure
        #figure.set_size_inches(24, 8)
        #figure.savefig('CLUSTER_ASSIGNATION.png', dpi=300)
        return clusters
        
        
clusters1 = DCplot(dist1, Y1, int(ND1), rho1, delta1, ordrho1, dc1, nneigh1, rhomin, deltamin)

dc_df = pd.DataFrame (data = Y1, columns = ['x', 'y'])
dc_df['cluster'] = clusters1[:,1]
dc_df['cluster_halo'] = clusters1[:,2]
dc_df['drug'] = tSNE_all[1]['drug']
dc_df['concentration'] = tSNE_all[1]['concentration']

sns.set_palette()
sns.lmplot(x= 'x', y='y', data=dc_df, hue = 'cluster', fit_reg = False)
plt.legend(loc=2, bbox_to_anchor=(1.05, 1) ,ncol = 1, frameon= True)
plt.tight_layout(rect=[0,0,0.75,1])


#with open('your_data.dat', 'w') as your_dat_file:  
 #   your_dat_file.write(temp)

#tSNE_all[21].to_csv(os.path.join(directoryA, 'test1.csv'), sep = ',', columns = ['SNE_1', 'SNE_2'],\
       # index_label=False, index=False, header = False)

#filein = os.path.join(directoryA, 'test1.csv')    
    
    
