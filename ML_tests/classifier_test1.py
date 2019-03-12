#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:27:02 2018

@author: ibarlow
"""

""" Analysis to use classifier to find the difference between Clozapine and all the other drugs

1. Make /Users/ibarlow/Documents/python scripts/Functions root directory

First going to load the features, filter them, and then start the classification

"""

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
fid_eggs = os.path.join (directoryA[:-7], 'egg_all.csv')
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

#%% now to do the stats on the experiments

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import feature_swarms as swarm

#make some violin plots of the significant features
cmap = sns.color_palette("tab20", len(uniqueDrugs))
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
    plt.figure()
    sns.swarmplot(x='drug', y=feature, data=features_df, \
                      hue = 'concentration', palette = cmap)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5) ,ncol = 1, frameon= True)    
    plt.xticks(rotation = 45)
    #plt.show()
    plt.savefig(os.path.join (directoryA[0:-7], 'Figures', rep1 + feature + file_type),\
                 bbox_inches="tight", dpi = 200) 

#for this it is usful to append the conditions onto the dataframe
for rep in featuresEA_1:
    featuresEA_1 [rep] ['drug'] = drugA2[rep]
    featuresEA_1[rep] ['concentration'] = concA2[rep]
    #featuresEA_1[rep]['exp'] =exp_namesA[rep]
    featuresEA_1[rep] ['date'] = dateA2[rep]
    
#compare each compound to control data
controlMeans = {}
for rep in featuresEA_1:
    controlMeans[rep] = pd.DataFrame()
    for line in range(len(featuresA2[rep])):
        if featuresEA_1[rep]['drug'].iloc[line] == 'DMSO':
            DMSO = featuresEA_1[rep].iloc[line]
            controlMeans[rep] = controlMeans[rep].append(DMSO.to_frame().transpose())
            del DMSO
    controlMeans[rep] = controlMeans[rep].reset_index(drop=True)

#so can now compile a matrix of p_values comparing each drug to control means
    #need to loop through the different drugs and basically just compare all the features to controlMeans
pVals = {}
feats =list(featuresEA_1[rep].columns)
feats = feats[0:-1]
for rep in featuresEA_1:
    pVals[rep] = pd.DataFrame()
    for drug in uniqueDrugs:
        if drug == 'DMSO':
            continue
        else:        
            currentInds = pd.DataFrame()
            for line in featuresEA_1[rep].iterrows():
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
    top_feats[rep]= pd.DataFrame(bh_p[rep].values[:,0:-2] <=0.05, columns = pVals[rep].iloc[:,0:-2].columns)
    post_exclude [rep]= []
    sig_feats[rep] = []
    for feat in range(bh_p[rep].shape[1]-2):
        if np.sum(top_feats[rep].iloc[:,feat]) ==0:
            post_exclude[rep].append(bh_p[rep].columns[feat])
        else:
            sig_feats[rep].append((bh_p[rep].columns [feat], np.sum(top_feats[rep].iloc[:,feat])))
    
    top_feats[rep]['drug'] = pVals[rep]['drug']
    top_feats[rep]['concentration'] = pVals[rep]['concentration']
    
    #sort by most to least
    sig_feats[rep].sort(key =lambda tup:tup[1])
    sig_feats[rep].reverse()

#filter out so only clozapine
cloz_feats = {}
cloz_list = pd.DataFrame()
for rep in top_feats:
    cloz_feats[rep] = top_feats[rep][top_feats[rep]['drug']=='Clozapine']
    cloz_list = cloz_list.append(np.sum(cloz_feats[rep] == True).to_frame().T)

#refine to features that are significant in at least three conditions
final = []
for item in cloz_list.columns:
    if np.sum(cloz_list[item]) > 3:
        final.append(item)
       
#this is a list with all the significant features that are different between clozapine and all the other conditions
sns.set_style('whitegrid')
swarms('all', 'curvature_hips_IQR', featMatAll, directoryA, '.tif')
    
#%% now to try and train a classifier - on raw data and PCA'd data
        
        #linear discriminant analysis + quadratic discriminant analysis
        #SVM?
            #first step is to do some features selection to refine the
            # features to use to train classifier
        
import numpy as np

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

#feature selection options: 
    # 1. LassoCV and Lasso method to determine cutoff for feature selection
    # 2. Linear SVC to recursive narrow down features
    # 3. Select from model to fit the data
    # 4. Transform


from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV, LassoLarsIC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

#make an array of the data
X =np.array (featMatAll.iloc[:,:-3])
y = np.array (drug_all)
drug_all2 = drug_all

drug_all2 = drug_all.transpose().to_frame()

drug_all3 = pd.DataFrame()
for row in range(0,drug_all2.shape[0]):
    row = drug_all2.iloc[row,:].to_frame().transpose()
    for drug in range(0, len(uniqueDrugs)):
        if (row['drug'] == uniqueDrugs[drug]).any():
            row['ID'] = int(drug)
            drug_all3 = drug_all3.append(row)
        else:
            continue
#drug_all3 contains indeces for the drugs
y = np.array(drug_all3['ID'])


#%% Lasso method           
    #use bayes information criteria to determine alpha
model_bic = LassoLarsIC(criterion='bic')
model_bic.fit(X, y)
alpha_bic_ = model_bic.alpha_

#or alternatively use akaike information criterion
model_aic = LassoLarsIC(criterion = 'aic')
model_aic.fit(X,y)
alpha_aic_ = model_aic.alpha_

def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information-criterion for model selection ')

#both converge on same answer of alpha  = 0.0628

#need to do cross-validation to check that the information criteria generated the correct result
# LassoCV: coordinate descent
# LassoLarsCV - least angle regression - better for examples where number of samples < no features
    #so do LassoLarsCV
    
import time as time

# Compute paths
print("Computing regularization path using the Lars lasso...")
t1 = time.time()
model = LassoLarsCV(cv=20).fit(X, y)
t_lasso_lars_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model.cv_alphas_)

#make figures
plt.figure()
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
          % t_lasso_lars_cv)
plt.axis('tight')

plt.show()

#okay so now can use the alpha number to apply the lasso method.
    #nb. higher alpha values are more stringent
    # I am going to use the Lars-determined alpha as it is less stringent, and 
    #literature says Lars is good for when no features > no samples, which I have

reg = Lasso(alpha = model.alpha_, max_iter = 10000,positive = True, selection = 'random')
reg.fit(X, y)
Lasso_features = reg.coef_
Lasso_select = Lasso_features>0
Lasso_feat_final = list(featMatAll.iloc[:,:-3].columns[Lasso_select])
#so 97 features picked out by Lasso method
    #lasso is a linear model selection method so may not be the best

#plot these
plt.figure()
plt.plot(Lasso_features)
plt.xlabel ('Features')
plt.ylabel ('Coefficient')
plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'Lasso_feature_selection.png'), dpi = 200)

del model, reg
#%% SVC method
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

lsvc = LinearSVC()
lsvc.fit(X,y)
model = SelectFromModel(lsvc, prefit = True)
X_new = model.transform(X)
X_new.shape
#reduce to 213 features - but what are these features?

plt.figure()
for drug in X_new:
    plt.plot(drug)

#make a dataframe of the new reduced feature set
SVC_features = pd.DataFrame (data= X_new)
SVC_features['drug'] = drug_all
SVC_features ['concentration' ] = conc_all  
#this may be good for using tSNE

import tSNE_custom as SNE
testing = np.arange(0,101,20)

SVC_SNE = {}
times = {}
for test in testing:
    SVC_SNE[test], times[test] = SNE.tSNE_custom(SVC_features,-2, testing)

for test in testing:
    SNE.sne_plot(SVC_SNE[test], testing, 10, uniqueConcs )

#doesn't seem to make that much difference to the tSNE

#lda on this set of features
X =  np.array(SVC_features.iloc[:,:-2])
y = np.array(drug_all)
clf = LinearDiscriminantAnalysis(n_components = None, priors = None, \
                                 shrinkage = 'auto', solver = 'eigen')
X2 = clf.fit_transform(X, y)

#convert class means into a dataframe
class_means = pd.DataFrame(data = clf.means_, columns = SVC_features.iloc[:,:-2].columns)
class_means['drug'] = clf.classes_

sns.set_style('whitegrid')
plt.figure()
for i in range(0,len(clf.means_)):
    plt.plot(clf.means_[i], label = clf.classes_[i])
plt.legend() 
plt.show()

del X, y
#%% do linear/quadratic discriminant analysis on reduced feature set generated from Lasso
    #because this may eventually be more interpretable
    

Lasso_features = pd.concat([featMatAll[Lasso_feat_final], featMatAll['concentration'], featMatAll['drug']], axis=1)
train = Lasso_features.iloc[:int(Lasso_features.shape[0]/2),:]
test = Lasso_features.iloc[int(Lasso_features.shape[0]/2):, :]

#linear discrimant test 1 (28 feb)
X =  np.array(train.iloc[:,:-2])
y = np.array(drug_all[:int(len(drug_all)/2)])
clf = LinearDiscriminantAnalysis(n_components = None, priors = None, \
                                 shrinkage = 'auto', solver = 'eigen')
X2 = clf.fit_transform(X, y)

X_test = np.array(test.iloc[:,:-2])
y_test = np.array(drug_all[int(len(drug_all)/2):])
clf.score(X_test,y_test)

#make a figure
plt.figure()
for drug in range(0,len(clf.means_)):
    plt.plot(clf.means_[drug], label = uniqueDrugs[drug])
plt.legend()
plt.show()

#okay so try alternative where just mark clozapine (10uM, and then the rest of the drugs are just 'other' - the )
drug_conc_all = pd.concat([drug_all, conc_all], axis =1)
for line in drug_conc_all.iterrows():
    if line[1]['drug'] == 'Clozapine':
        if line[1]['concentration'] == 10:
            line[1]['drug'] = 'Clozapine10'
        else:
            line[1] ['drug'] == 'Clozapine'
    elif line[1]['drug'] == 'DMSO' or line[1]['drug'] == 'No_compound': 
        line[1]['drug'] = line[1]['drug']
    else:
        line[1]['drug'] = 'Other'   
      

#redo classification
y2= np.array(drug_conc_all['drug'].iloc[:int(len(drug_conc_all['drug'])/2)])
X3 = clf.fit_transform(X, y2) 

y_test2 = np.array(drug_conc_all['drug'].iloc[int(len(drug_conc_all['drug'])/2):])
score_1 = clf.score(X_test, y_test2)

#make figure:
sns.set_style('whitegrid')
sns.set_palette(sns.color_palette("tab20", clf.means_.shape[0]))
plt.figure()
for drug in range(0,len(clf.means_)):
    plt.plot(clf.means_[drug], label = clf.classes_[drug])
#plt.xticks(range(1,len(clf.means_[drug])+1),Lasso_features.iloc[:,:-2].columns, rotation = 45)
plt.legend()
plt.xlabel('Features')
plt.ylabel ('Mean score')
plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'lasso_figures_mean_scores.png'), dpi= 200)
plt.show()

#make this into a dataframe
Cloz_LD = pd.DataFrame(data = clf.means_, columns = Lasso_features.iloc[:,:-2].columns)
Cloz_LD['drug'] = pd.Series(clf.classes_)

#sort the class_mean values - ascending
cloz_only = Cloz_LD[Cloz_LD['drug']=='Clozapine10'].iloc[:,:-2]
cloz_sorted = pd.DataFrame(stats.rankdata(cloz_only)).transpose()
cloz_sorted.columns = Cloz_LD.iloc[:,:-2].columns
#make a list of what the features are
cloz_list2 = cloz_sorted.sort_values([0], axis=1)
cloz_list2 = np.flip(list(cloz_list2.columns), axis = 0)

cloz_only2 = cloz_only[cloz_list2]

sum(np.sum(cloz_only2>0)) #48 features that distinguish clozapine 10 from the other drugs

#plot these sorted values
    #first make dataframe and add on the descriptors
cloz_sort_df = Cloz_LD[cloz_list2]
cloz_sort_df['drug'] = Cloz_LD['drug']

#make the figure
plt.figure()
plt.plot(np.ones(97), 'm--', alpha = 0.4) #upper threshold
plt.plot(-1*np.ones(97), 'm--', alpha = 0.4) #lower threshold
for line in cloz_sort_df.iterrows():
    plt.plot(np.array(line[1][:-1]), label = line[1]['drug'])
plt.xlabel('Features')
plt.ylabel ('Mean score')
plt.legend()
plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'sorted_lasso_features.png'), dpi = 200)
plt.show()

#now select these top ones
top = [int(i) for i in list(np.argwhere(np.sum(cloz_only2 > 1.0)))]
bottom = [int(i) for i in list(np.argwhere(np.sum(cloz_only2<-1.0)))]
final_cloz = pd.concat([cloz_sort_df.iloc[:,top], cloz_sort_df.iloc[:,bottom], cloz_sort_df.iloc[:,-1]], axis = 1)

plt.figure()
for line in final_cloz.iterrows():
    plt.plot(np.array(line[1][:-1]), label = line[1]['drug'])
plt.xticks(range(0,final_cloz.iloc[:,:-1].shape[1]),final_cloz.iloc[:,:-1].columns, rotation = 45)
plt.legend()
plt.show()

#final final features
final_LassoLD = list(final_cloz.iloc[:,:-1].columns)

#make plots
for rep in reps:
    for item in final_LassoLD:
        swarms(rep, item, featuresEA_1[rep], directoryA, '.tif' )
        plt.close()

#make some plots with the combined data
for item in final_LassoLD:
    swarms('all', item, featMatAll, directoryA, '.svg' )
    plt.close()

swarms('all', 'eggs', featMatAll, directoryA, '.svg')

#try the clustermap
cloz_cluster = featMatAll[final_cloz.columns]
    
#make lut for drug colors
lut = dict(zip(uniqueDrugs, cmap))
row_colors = featMatAll['drug'].map(lut)#map onto the feature Matrix

#make clustergram
cg=sns.clustermap(featMatAll[final_cloz.columns].iloc[:,:-1], metric  = 'euclidean', cmap = 'inferno', row_colors = row_colors)
order = list(featMatAll[final_cloz.columns].iloc[cg.dendrogram_row.reordered_ind, -1])
#plt.setp(cg.ax_heatmap.yaxis.set_ticks(order, minor = True))
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels(featMatAll[final_cloz.columns].iloc[cg.dendrogram_row.reordered_ind, -1]))
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels(featMatAll[final_cloz.columns].iloc[cg.dendrogram_row.reordered_ind, -1]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8)
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 45, fontsize = 10) 
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'Cloz_LassoLD.svg'), dpi =150, bbox_inches = 'tight', pad_inches = 1)

colors = [(v) for k,v in lut.items()]
plt.figure()
ax = plt.imshow([colors])
ax.axes.set_xticklabels(lut.keys(), rotation = 45)
ax.axes.xaxis.set_ticks_position('top')
ax.xlim = [-1, 10]
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'drug_colors.png'), dpi =150)

#%% also try running LDA without feature selection

#linear discrimant test 1 (28 feb)
del X,y,X2, X_test, y_test
train2 = featMatAll.iloc[:int(featMatAll.shape[0]/2),:]
test2 = featMatAll.iloc[int(featMatAll.shape[0]/2):,:]

conds2 = pd.concat([drug_all, conc_all], axis =1)
#make this condition so just clozapine at 10um vs all the others - 2 conditions
for line in conds2.iterrows():
    if line[1]['drug'] == 'Clozapine' and line[1]['concentration'] == 10:
        line[1]['drug'] = 'Clozapine10'
    else:
        line[1] ['drug'] ='Other'
    #else:
     #   line[1]['drug'] = 'Other' 

X =  np.array(train2.iloc[:,:-3])
y = np.array(conds2.iloc[:int(conds2.shape[0]/2),:]['drug'])
clf = LinearDiscriminantAnalysis(n_components = None, priors = None, \
                                 shrinkage = 'auto', solver = 'eigen')
X2 = clf.fit_transform(X, y)

X_test = np.array(test2.iloc[:,:-3])
y_test = np.array(conds2.iloc[int(conds2.shape[0]/2):,:]['drug'])
clf.score(X_test,y_test)

#make a figure
plt.figure()
for drug in range(0,len(clf.means_)):
    plt.plot(clf.means_[drug], label = clf.classes_[drug])
plt.legend()
plt.show()

#make this into a dataframe
Cloz_LD2 = pd.DataFrame(data = clf.means_, columns = featMatAll.iloc[:,:-3].columns)
Cloz_LD2['drug'] = pd.Series(clf.classes_)

#sort the class_mean values - ascending
cloz_only2 = Cloz_LD2[Cloz_LD2['drug']=='Clozapine10'].iloc[:,:-1]
cloz_sorted2 = pd.DataFrame(stats.rankdata(cloz_only2)).transpose()
cloz_sorted2.columns = Cloz_LD2.iloc[:,:-1].columns
#make a list of what the features are
cloz_list2 = cloz_sorted2.sort_values([0], axis=1)
cloz_list2 = np.flip(list(cloz_list2.columns), axis = 0)

cloz_only2_2 = cloz_only2[cloz_list2]

sum(np.sum(cloz_only2>0)) #48 features that distinguish clozapine 10 from the other drugs

#plot these sorted values
    #first make dataframe and add on the descriptors
cloz_sort_df2 = Cloz_LD2[cloz_list2]
cloz_sort_df2['drug'] = Cloz_LD2['drug']

#make the figure
plt.figure()
plt.plot(np.ones(500), 'm--', alpha = 0.4) #upper threshold
plt.plot(-1*np.ones(500), 'm--', alpha = 0.4) #lower threshold
for line in cloz_sort_df2.iterrows():
    plt.plot(np.array(line[1][:-1]), label = line[1]['drug'])
plt.xlabel('Features')
plt.ylabel ('Mean score')
plt.legend()
plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'sorted_lasso_features.png'), dpi = 200)
plt.show()

#now select these top ones
top = [int(i) for i in list(np.argwhere(np.sum(cloz_only2_2 > 1.0)))]
bottom = [int(i) for i in list(np.argwhere(np.sum(cloz_only2_2<-1.0)))]
final_cloz = pd.concat([cloz_sort_df2.iloc[:,top], cloz_sort_df2.iloc[:,bottom], cloz_sort_df2.iloc[:,-1]], axis = 1)

plt.figure()
for line in final_cloz.iterrows():
    plt.plot(np.array(line[1][:-1]), label = line[1]['drug'])
plt.xticks(range(0,final_cloz.iloc[:,:-1].shape[1]),final_cloz.iloc[:,:-1].columns, rotation = 45)
plt.legend()
plt.show()

#final final features
final_LassoLD = list(final_cloz.iloc[:,:-1].columns)

#make some plots with the combined data
for item in final_LassoLD:
    swarms('all', item, featMatAll, directoryA, '.tif' )
    plt.close()




#%%Junk from here on
    
    
#try clustergrammer 
from clustergrammer import Network
net = Network()
net.load_df(cloz_cluster)

# filter for the top 100 columns based on their absolute value sum
net.filter_N_top('col', 16, 'sum')

# cluster using default parameters
net.cluster()

#leg_patch = mpatches.Patch(color = cmap, label=[lut.keys()])
#plt.legend(handles=[leg_patch])

plt.legend ([cmap], lut.keys())
plt.show()
x
swarms (rep1, feature, features_df, directory, file_type)

testX = np.array(featuresZ2[reps[0]].iloc[:,:-3])
testy = np.array(drugA2[reps[0]])
clf.score(testX, testy)

#using training set as reps[0] and then testing with the other two experiments,
 #only getting classification efficientcy of 0.43 - 0.47

#need to find the set of features that best separate clozapine (10um concentration) from the other drugs
 #use the entire dataset
     #pull out from the lda which features contribute to the classes
del X,y ,X2

X = np.array(featMatAll.iloc[:,:-3])
y = np.array(drug_all)
#use same clf
X2 = clf.fit_transform(X,y)

#convert class means into a dataframe
class_means = pd.DataFrame(data = clf.means_, columns = Lasso_features.iloc[:,:-2].columns)
class_means['drug'] = clf.classes_

sns.set_style('whitegrid')
plt.figure()
for i in range(0,len(clf.means_)):
    plt.plot(clf.means_[i], label = clf.classes_[i])
plt.legend() 
plt.show()

#just plot the sorted values first to check out the shape of the curves for the different drugs
for i in range(0, len(clf.means_)):
    plt.plot(np.sort(clf.means_[i]), label = clf.classes_[i])
plt.legend()
plt.show()

#sort the class_mean values - ascending
cloz_only = class_means[class_means['drug']=='Clozapine'].iloc[:,:-1]
cloz_sorted = np.argsort(class_means[class_means['drug']=='Clozapine'].iloc[:,:-1])
cloz_sorted = cloz_sorted.reset_index(drop =True)
#make a list of what the features are
cloz_list2 = cloz_sorted.sort_values([0], axis=1)
cloz_list2 = list(cloz_list2.columns)

#------------------------------------------------------------------
#next thing to try is to run classifier with 'clozapine-10' and then all the rest as 'other'
    #to train it just to classify cloz-10 vs the rest

    #set priors for QDA according to proportion of data - this may help!

sns.set_style('whitegrid')
plt.figure()
for i in range(0,len(clf.means_)):
    plt.plot(temp.iloc[i,:-1], label = temp['drug'].iloc[i])
plt.legend() 
plt.show()


#top 50
temp1 = class_means[cloz_list2[-50:]]

#this is an index for then sorting all the acual weighting values
class_sorted = {}
for drug in class_means

#this is still very noisy to look at
    #subtract the average of all the other means from clozapine to be left with only the clozapine peaks and trought/...?
clozapine= class_means[class_means['drug']=='Clozapine'].iloc[:,:-1] -class_means[class_means['drug']!='Clozapine'].iloc[:,:-1].mean(axis=0)






clf2 = QuadraticDiscriminantAnalysis(priors = y, store_covariance = True)
clf2.fit(X,y)




#try mRMR on featMatAll
import pymrmr as mrmr
