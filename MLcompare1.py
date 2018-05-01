#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:13:33 2018

@author: ibarlow
"""

""" New script to do machine learning of Clozapine-specific dataset

1. Load the features:
    a) Plate summaries(including the Eggs dataset),
    b) Individual tracks 
    
2. Eliminant nan features, standardise (z-score)

3. Perform LDA on entire feature set to separate clozapine (all doses) vs other
    a) train, cross-validation, and test sets
    b) compare error or training and CV sets to determine if need to reduce feature set size

4. Feature selection (using mRMR) if necessary

5. Retrain on reduced feature set

6. What is the training score for this reduced feature set?

7. Clustering

Required functions
TierPsyInput.py
feature_swarms.py

"""

#First load the plate summary features
    #set directory of functions
cd Documents/pythonScripts/Functions/
import TierPsyInput as TP
import numpy as np
import pandas as pd
import os

#load plate stats
directoryP, fileDirP, featuresP =  TP.TierPsyInput('new', 'Liquid')

#load trajectories
directoryT, fileDirT, featuresT = TP.TierPsyInput('old', 'Liquid')

#first filter out tracks less <3000 frames from featuresT
#minimum trajectory length                
minLength = 3000;
featuresT2 = {}
for rep in featuresT:
    featuresT2[rep]=featuresT[rep][featuresT[rep]['n_frames']>=minLength]


#from here pull out info about each of the experiment's variables, such as drug, concentration and uniqueID
    #first, remove experiment list
exp_namesP={}
exp_namesT = {}
featuresP2 = featuresP.copy()
for rep in featuresP:
    exp_namesP[rep] = featuresP[rep].pop('exp')
    exp_namesT[rep] = featuresT2[rep].pop('exp')
    
    
#initial dictionaries
drugP = {}
drugT = {}
concP = {}
concT ={}
dateP = {}
dateT = {}
uniqueIDP = {}
uniqueIDT = {}
for rep in exp_namesP:
    drugP[rep], concP[rep], dateP[rep], uniqueIDP[rep] = TP.extractVars(exp_namesP[rep])
    drugT[rep], concT[rep], dateT[rep], uniqueIDT[rep] = TP.extractVars(exp_namesT[rep])

#make lists of unqiue drugs and concs
drugs = []
concs = []
dates =[]
for rep in drugP:
    drugs.append(list(drugP[rep].values))
    concs.append(list(concP[rep].values))
    dates.append(list(dateP[rep].values))
flatten = lambda LoL: [item for sublist in LoL for item in sublist] #lambda function flattens list of lists (LoL)

uniqueDrugs = list(np.unique(flatten(drugs)))
uniqueConcs = list(np.unique(flatten(concs)))
uniqueDates = list(np.unique(flatten(dates)))
del drugs, concs, dates   

#%% import the eggs data to append to featuresP
#fileid for eggs data
fid_eggs = os.path.join (directoryP[:-7], 'ExtraFiles', 'egg_all.csv')
#load data
eggs_df = pd.read_csv(fid_eggs)

#split the data by experiment
eggs={}
reps = list(fileDirP.keys())

#make dictionary to match rep and date
rep_match = {}
for rep in dateP:
    rep_match[list(np.unique(dateP[rep].values))[0]] = rep

#and then compile dictionary using this   
for date in uniqueDates:
    eggs[rep_match[date]] = eggs_df[eggs_df['date'] ==int(date)]
    eggs[rep_match[date]] = eggs[rep_match[date]].reset_index(drop=True)

del rep_match, reps
#problem is that there is not egg data for every plate... so need to match up data accordingly

#add on descriptors to A2 dataframe
featuresP3 = {}
for rep in featuresP2:
    featuresP3[rep] = pd.concat([featuresP2[rep], dateP[rep], drugP[rep], concP[rep], uniqueIDP[rep]], axis =1)

#now to match them up
eggs2 = eggs.copy()
eggs_df2 = {}
for rep in featuresP3:
    eggs_df2[rep] = pd.DataFrame()
    for step in range(0, featuresP3[rep].shape[0]):
        line = featuresP3[rep].iloc[step,:]
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
featuresEP1 = eggs_df2.copy()
    
del eggs2,featuresP3

#need to pop out the experiment descriptors again as may be wrong
drugP2 = {}
concP2 ={}
dateP2={}
uniqueIDP2 = {}
for rep in featuresEP1:
    drugP2[rep] = featuresEP1[rep].pop('drug')
    concP2[rep] = featuresEP1[rep].pop('concentration')
    dateP2[rep] = featuresEP1[rep].pop('date')
    uniqueIDP2[rep] = featuresEP1[rep].pop('uniqueID')
  
#%% can now go on to standardise and filter the features

#filter features in tracks for nans
to_excludeP={}
to_excludeT={}
for rep in featuresEP1:
    to_excludeP[rep] = TP.FeatFilter(featuresEP1[rep])
    to_excludeT[rep] = TP.FeatFilter(featuresT2[rep])
    
#combined for all experiments to exclude
list_excludeP = [y for v in to_excludeP.values() for y in v]
list_excludeP = list(np.unique(list_excludeP))

list_excludeT = [y for v in to_excludeT.values() for y in v]
list_excludeT = list(np.unique(list_excludeT))

#list_exclude P is empty
    #remove features from list_excludeT
featuresT3 = {}
for rep in featuresT2:
    featuresT3[rep] = TP.FeatRemove(featuresT2[rep], list_excludeT)

# Z-score normalisation
featuresZP={}
featuresZT = {}
for rep in featuresEP1:
    featuresZP[rep] = TP.z_score(featuresEP1[rep])
    featuresZT[rep] = TP.z_score(featuresT3[rep])

#double check nan features have been removed
to_excludeZP={}
for rep in featuresZP:
    to_excludeZP[rep] = TP.FeatFilter(featuresZP[rep])
    
#combined for all experiments to exclude
list_excludeZP = [y for v in to_excludeZP.values() for y in v]
list_exclude = list(np.unique(list_excludeZP))

#remove these features from the features dataframes
for rep in featuresZP:
    featuresZP[rep] = TP.FeatRemove(featuresZP[rep], list_excludeZP)  

#Fill the nans in both data sets
featMatP = {}
featMatT = {}
for rep in featuresZP:
    featMatP[rep] = featuresZP[rep].fillna(featuresZP[rep].mean(axis=0))
    featMatT[rep] = featuresZT[rep].fillna(featuresZT[rep].mean(axis=0))

#and now concatenate all into one large datafram with descriptors added in
featMatAllP = pd.DataFrame()
featMatAllT = pd.DataFrame()
for rep in featMatP:
    featMatAllP = featMatAllP.append(pd.concat([featMatP[rep], drugP2[rep], concP2[rep],\
                                                dateP2[rep]], axis = 1))
    featMatAllT = featMatAllT.append(pd.concat([featMatT[rep], drugT[rep], concT[rep],\
                                                dateT[rep]], axis=1))

#reset indecis
featMatAllP = featMatAllP.reset_index( drop= True, inplace=False)
featMatAllT = featMatAllT.reset_index(drop = True, inplace= False)
    
#final featMatAllT  should not include worm index, n_frames, is_good_skel, first_frame so remove these
featMatAllT.drop(['worm_index', 'n_frames', 'n_valid_skel', 'first_frame'], axis= 1, inplace = True ) 

#%% 
""" implementation of LDA
1. Make a conditions dataframe containing the descriptors of the data
2. Use Stratified Shuffle split cross validation
3. 

 """

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

condsT = featMatAllT.iloc[:,-3:]
condsP = featMatAllP.iloc[:,-3:]

#make another dataframe for clozapine10 vs all others
condsT2 = pd.DataFrame()
for line in condsT.iterrows():
    line2 = line[1].to_frame().transpose()
    if line[1]['drug'] == 'Clozapine' and line[1]['concentration'] == 10:
        line2['drug'] = 'Clozapine10'
        line2['class'] = 'Clozapine10'
        line2['class2'] = 1
    else:
        line2 ['drug'] =line[1]['drug']
        line2['class'] = 'Other'
        line2['class2'] = 0
    condsT2 = condsT2.append(line2)
    del line2
condsT2 = condsT2.reset_index(drop =True, inplace=False)

#and for plate data
condsP2 = pd.DataFrame()
for line in condsP.iterrows():
    line2 = line[1].to_frame().transpose()
    if line[1]['drug'] == 'Clozapine' and line[1]['concentration'] == 10:
        line2['drug'] = 'Clozapine10'
        line2['class'] = 'Clozapine10'
        line2['class2'] = 1
    else:
        line2 ['drug'] =line[1]['drug']
        line2['class'] = 'Other'
        line2['class2'] = 0
    condsP2 = condsP2.append(line2)
    del line2
condsP2 = condsP2.reset_index(drop=True, inplace=False)

#use label encoder to make drugs numerical - this will be useful later for mRMR
#train encoder
drugLabelsT = condsT['drug'].values
enc = LabelEncoder()
label_encoder = enc.fit(drugLabelsT)
drugNumsT = label_encoder.transform(drugLabelsT) + 1

#append into the dataframe
condsT2['drug2'] = drugNumsT
del label_encoder

drugLabelsP = condsP['drug'].values
label_encoder = enc.fit(drugLabelsP)
drugNumsP = label_encoder.transform(drugLabelsP)+ 1

condsP2['drug2'] = drugNumsP

del drugNumsP, drugNumsT, enc, label_encoder

#now ready to train classifier
#set data (X) and labels (y)
Xp = np.array(featMatAllP.iloc[:,:-3])
yp = condsP2['class']

Xt = np.array(featMatAllT.iloc[:,:-3])
yt = condsT2['class']

PriorP = [sum(condsP2['class']=='Clozapine10')/len(condsP2), sum(condsP2['class'] == 'Other')/len(condsP2)]
PriorT = [sum(condsT2['class'] == 'Other')/len(condsT2), sum(condsT2['class'] == 'Clozapine10')/len(condsT2)]


#for splitting the data and crossvalidation
    #shuffle 10-fold cross validation
sss = StratifiedShuffleSplit(n_splits = 10, train_size=0.6)

#skf = StratifiedKFold (n_splits = 3, shuffle=True)

#set liniar discriminant classifier
clf = LinearDiscriminantAnalysis(n_components = None, priors = PriorP, \
                                 shrinkage = None, solver = 'svd', tol = 1e-6)
import random
#cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3)
scoresT = []
for i in range (1,300,1):
    featRand = random.sample(list(featMatAllT.iloc[:,:-3].columns), i)

    Xrand = np.array(featMatAllT[featRand])
    scoresT.append(cross_val_score(clf, Xrand, yt, cv=sss))
    del Xrand, featRand

#make an array of score averages to plot
scoreTav = []
scoreTstd=[]
scoreTsem = []
for featNo in scoresT:
    scoreTav.append(np.mean(featNo))
    scoreTstd.append(np.std(featNo))
    #scoreRsem.append(np.std(scoreRav[featNo]/np.sqrt(len(scoreRav[featNo][0]))))
    
scoresP = []
for i in range (1,300,1):
    featRand = random.sample(list(featMatAllP.iloc[:,:-3].columns), i)
    
    Xrand = np.array(featMatAllP[featRand])
    scoresP.append(cross_val_score(clf, Xrand, yp, cv=sss))
    del featRand, Xrand

#make an array of score averages to plot
scorePav = []
scorePstd=[]
scoreRsem = []
for featNo in scoresP:
    scorePav.append(np.mean(featNo))
    scorePstd.append(np.std(featNo))
    #scoreRsem.append(np.std(scoreRav[featNo]/np.sqrt(len(scoreRav[featNo][0]))))
sns.set_style ('whitegrid')
plt.plot(scorePav) 
plt.plot(scoreTav, 'r')
plt.xlabel('number of features')
plt.ylabel('classification score')

#%% Apply mRMR to both feature sets

import pymrmr
import matplotlib.pyplot as plt

#need to discretise the data prior to implementing the algorithm
        # use pandas cut function - so bin data using
        #The Freedman-Diaconis Rule says that the optimal bin size of a histogram is
        # Bin Size=2⋅IQR(x)n^(−1/3) and then divide this by the total range of 
        # the data to determine the number of bins

#use histogram blocks - freedman diaconis
binCutoffP = {}
binCutoffT = {}
#plate features
for feat in featMatAllP.iloc[:,:-3].columns:
    plt.ioff()
    binCutoffP[feat] = np.histogram(featMatAllP[feat], bins='fd')[1]
#trajectories features
for feat in featMatAllT.iloc[:,:-3].columns:
    binCutoffT[feat] = np.histogram(featMatAllT[feat], bins= 'fd')[1]

#use these to create bins for the cutting up the data - can input into pandas cut
catP = pd.DataFrame()
catT = pd.DataFrame()
for feat in featMatAllP.iloc[:,:-3].columns:
    catP[feat]=pd.cut(featMatAllP[feat], bins= binCutoffP[feat], \
       labels = np.arange(1,len(binCutoffP[feat])), include_lowest=True)
#trajectories data
for feat in featMatAllT.iloc[:,:-3].columns:
    catT[feat]=pd.cut(featMatAllT[feat], bins = binCutoffT[feat],\
        labels = np.arange(1,len(binCutoffT[feat])), include_lowest = True)

#make ints
catP = pd.DataFrame(data = np.array(catP.values), dtype = int, columns = catP.columns)        
catT = pd.DataFrame(data= np.array(catT.values), dtype = int, columns = catT.columns)

#add in info about rows
catP.insert(0, column = 'class', value = condsP2['class2'].values, allow_duplicates=True)
catT.insert(0, column = 'class', value = condsT2['class2'].values, allow_duplicates=True)

#select 300 features using mRMR
    #based on all drugs
mrFeatsP = pymrmr.mRMR(catP, 'MID', 200)
mrFeatsT = pymrmr.mRMR(catT, 'MID', 300)

mrFeatMatAllP = pd.concat([featMatAllP[mrFeatsP], condsP2], axis=1)

#export these features as txt tile
out = open(os.path.join(directoryP[:-7], 'mRMR_featsAgarP.txt'), 'w')
out.writelines(["%s\n" % item  for item in mrFeatsP])
out.close()


#find classification erro of the mRMR features
scoresPmRMR = []
for i in range (1,201,1):
    XmR = np.array(mrFeatMatAllP.iloc[:,:i])
    scoresPmRMR.append(cross_val_score(clf, XmR, yp, cv=sss))
    del XmR

#make an array of score averages to plot
scorePMRav = []
scorePMRstd=[]
scoreRPMsem = []
for featNo in scoresPmRMR:
    scorePMRav.append(np.median(featNo))
    scorePMRstd.append(np.std(featNo))
    #scoreRsem.append(np.std(scoreRav[featNo]/np.sqrt(len(scoreRav[featNo][0]))))
sns.set_style ('whitegrid')
plt.plot(scorePMRav) 
plt.plot(scoreTav, 'r')
plt.xlabel('number of features')
plt.ylabel('classification score')

#make a colormap to assign colours - based on class (ie clozapine10 is separate)
cmap1 = sns.color_palette("tab20", len(np.unique(mrFeatMatAllP['drug'])))

#make a clustergram
    #1. make lut for drug colors
    #2. map the lut onto the clustergram
lut = dict(zip(np.unique(mrFeatMatAllP['drug']), cmap1))

#add in row colors to the dataframe

row_colors = mrFeatMatAllP['drug'].map(lut)#map onto the feature Matrix

#make clustergram
cg=sns.clustermap(mrFeatMatAllP.iloc[:,:27], metric  = 'euclidean', cmap = 'inferno', \
                  row_colors = row_colors)
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
         (mrFeatMatAllP['drug'][cg.dendrogram_row.reordered_ind]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8) #y tick labels
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) #x tick labels
#set position
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
#save fig
plt.savefig(os.path.join(directoryTemp[0:-7], 'Figures', 'AgarLDA_clustergramMRMR' + str(featNo) + '.tif'), \
            dpi =150, bbox_inches = 'tight', pad_inches = 1)
plt.show()




#TO DO:
#2. Practice introduction etc in the morning

#%% find the features that correlate most highly with Clozapine and clozpine 10

#make big data frame with drug10 descriptors as well
featMatAllP2 = pd.concat([featMatAllP.iloc[:,:-3], condsP2['drug'], condsP2['concentration']], axis =1)

#now make dummy dataframe containg indexes for the different conditions
dummyP = pd.get_dummies(featMatAllP2['drug'])

#concat with featMatAllP and then do correlation with clozapine10

featMatAllPCorr = pd.concat([featMatAllP2.iloc[:,:-2], dummyP], axis=1)

CorrP = featMatAllPCorr.corr()['Clozapine10'].sort_values()

testDf = featMatAllP2[CorrP.keys()[-50:-1]]

testDf['drug'] = featMatAllP2['drug']

row_colors = mrFeatMatAllP['drug'].map(lut)

#make clustergram
cg=sns.clustermap(testDf.iloc[:,:-1], metric  = 'euclidean', cmap = 'inferno', \
                  row_colors = row_colors)
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
         (mrFeatMatAllP['drug'][cg.dendrogram_row.reordered_ind]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8) #y tick labels
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) #x tick labels
#set position
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])


#%% another test using affinity propagation
from sklearn.cluster import AffinityPropagation

test = 
af = AffinityPropagation()
