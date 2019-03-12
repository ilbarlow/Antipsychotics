#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:32:23 2018

@author: ibarlow
"""

""" Script to test out using mRMR and then using LDA or QDA classifiers:
Objective:
    1. define feature set that is unique to Clozapine - 10uM
    2. use for hierachical clustering - can this pick out the 10uM Clozapine cluster?
    3. 
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
    
#%% combine the data

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

#%% pymRMR

""" pymrmr is not a standard library in pip or conda; instead install from git page:
    In the terminal:
        brew install cython
        brew install llvm
        
        #then this links the llvm binaries so that pymrmr can find these compilers when
        running installation
        ln -s /usr/local/opt/llvm/bin/clang /usr/local/bin/clang-omp
        ln -s /usr/local/opt/llvm/bin/clang++ /usr/local/bin/clang-omp++
        
        #clone the pymrmr repository
        git clone https://github.com/fbrundu/pymrmr
    
    
    To setup installation need to go into downloaded pymrmr folder and then make sure that the setup.py
    in there can see the cython and clang compiler.
    
    Inside the setup.py script, need to add in these two lines at 29 and 30
     os.environ['LDFLAGS'] = "-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
     os.environ['CPPFLAGS'] = '-I/usr/local/opt/llvm/include'
  
    #then type in the terminal
    cd
    ls
    cd repositories/pymrmr
    python setup.py develop
    
    nb.
    This should work!"""

import pymrmr
from scipy import stats
import matplotlib.pyplot as plt

#need to discretise the data prior to implementing the algorithm
    #kernel density function can do this
        #alternative is to use pandas cut function - so bin data using
        #The Freedman-Diaconis Rule says that the optimal bin size of a histogram is
        # Bin Size=2⋅IQR(x)n^(−1/3)
        #and then divide this by the total range of the data to determine the number of bins

#use histogram blocks - freedman diaconis
#use bayesian blocks to partition the data
#from astroML.plotting import hist as block_hist
bin_cutoff = {}
for feat in featMatAll2.columns:
    plt.ioff()
    bin_cutoff[feat] = np.histogram(featMatAll2[feat], bins='fd')[1]

#use these to create bins for the cutting up the data - can input into pandas cut
cat = pd.DataFrame()
for feat in featMatAll2.columns:
    cat[feat]=pd.cut(featMatAll[feat], bins= bin_cutoff[feat], \
       labels = np.arange(1,len(bin_cutoff[feat])), include_lowest=True)

#make ints
cat2 = pd.DataFrame(data = np.array(cat.values), dtype = int, columns = cat.columns)        
#add in info about rows
cat.insert(0, column = 'drug', value = featMatAll['drug'], allow_duplicates=True)


#select 150 features using mRMR
mr_Feats = np.array(pymrmr.mRMR(cat2, 'MID', 150))

#export these features as txt tile
out = open(os.path.join(directoryA[:-7], 'mRMR_feats.txt'), 'w')
out.writelines(["%s\n" % item  for item in mr_Feats])
out.close()

del bin_cutoff, date, rep
#%%
#try LDA on this reduced feature set

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
QuadraticDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
import feature_swarms as swarm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set_style ('whitegrid')

#import mRMR features again
with open(os.path.join(directoryA[:-7], 'mRMR_feats.txt')) as f:
    mr_Feats = [line.rstrip() for line in f]

with open(os.path.join(directoryA[:-7], 'mRMR_feats2.txt')) as f:
    mr_Feats2 = [line.rstrip() for line in f]

#compare the two runs of mRMR - first one done with bayesian block and the\
# second with freedman diaconis
both = list(set(mr_Feats).intersection(mr_Feats2))
excl = list(set(mr_Feats).difference(mr_Feats2))

#make reduced feature set classifier
mrFeatMatFinal = pd.concat([featMatAll2[mr_Feats2], featMatAll.iloc[:,-3:]], axis = 1)

#shuffle the data
mrFeatMatFinal['shuffle'] = np.random.random_sample(size=221)
mrFeatMatFinal2 = mrFeatMatFinal.sort_values(by=['shuffle'])

#separate into training and testing set - 50% 50%
train = mrFeatMatFinal2.iloc[:int(mrFeatMatFinal2.shape[0]/2),:]
test = mrFeatMatFinal2.iloc[int(mrFeatMatFinal2.shape[0]/2):, :]

#make a conditions dataframe
conds = pd.concat([mrFeatMatFinal2['drug'], mrFeatMatFinal2['concentration']], axis =1)

#make this condition so just clozapine at 10um vs all the others - 2 conditions
conds2 = conds.copy()
for line in conds2.iterrows():
    if line[1]['drug'] == 'Clozapine' and line[1]['concentration'] == 10:
        line[1]['drug'] = 'Clozapine10'
    else:
        line[1] ['drug'] ='Other'
 
#now train LDA
X =  np.array(train.iloc[:,:-4]) #array of values
y = np.array(conds2.iloc[:int(conds2.shape[0]/2),:]['drug']) #array of conditions
#there are only 4 Clozapine 10uM samples in training set

clf = LinearDiscriminantAnalysis(n_components = None, priors = None, \
                                 shrinkage = 'auto', solver = 'eigen')
X2 = clf.fit_transform(X, y)

#test data
X_test = np.array(test.iloc[:,:-4])
y_test = np.array(conds2.iloc[int(conds2.shape[0]/2):,:]['drug'])
score = clf.score(X_test,y_test)
print (score *100)

#make a figure
plt.figure()
for drug in range(0,len(clf.means_)):
    plt.plot(clf.means_[drug], label = clf.classes_[drug])
plt.legend()
plt.show()

#make this into a dataframe
Cloz_LD = pd.DataFrame(data = clf.means_, columns = mrFeatMatFinal2.iloc[:,:-4].columns)
Cloz_LD['drug'] = pd.Series(clf.classes_) #attach on the classification names

#sort the class_mean values - ascending
cloz_only = Cloz_LD[Cloz_LD['drug']=='Clozapine10'].iloc[:,:-1]
cloz_sorted = pd.DataFrame(stats.rankdata(cloz_only)).transpose()
cloz_sorted.columns = Cloz_LD.iloc[:,:-1].columns
#make a list of what the features are
cloz_list = cloz_sorted.sort_values([0], axis=1)
cloz_list = np.flip(list(cloz_list.columns), axis = 0)

#sort dataframe by cloz_list
cloz_only2 = cloz_only[cloz_list]

#plot these sorted values
    #first make dataframe and add on the descriptors
cloz_sort_df = Cloz_LD[cloz_list]
cloz_sort_df['drug'] = Cloz_LD['drug']

#make the figure
plt.figure()
plt.plot(1*np.ones(150), 'm--', alpha = 0.4) #upper threshold
plt.plot(-1*np.ones(150), 'm--', alpha = 0.4) #lower threshold
for line in cloz_sort_df.iterrows():
    plt.plot(np.array(line[1][:-1]), label = line[1]['drug'])
plt.xlabel('Features')
plt.ylabel ('Mean score')
plt.legend()
plt.savefig(os.path.join(directoryA[:-7], 'Figures', 'sorted_mRMR_LDA_feats.png'), dpi = 200)
plt.show()

#now select these top ones
top = [int(i) for i in list(np.argwhere(np.sum(cloz_only2 > 1)))]
bottom = [int(i) for i in list(np.argwhere(np.sum(cloz_only2<-1)))]
final_cloz = pd.concat([cloz_sort_df.iloc[:,top], cloz_sort_df.iloc[:,bottom],\
                        cloz_sort_df.iloc[:,-1]], axis = 1)

    #make a figure
plt.figure()
for line in final_cloz.iterrows():
    plt.plot(np.array(line[1][:-1]), label = line[1]['drug'])
plt.xticks(range(0,final_cloz.iloc[:,:-1].shape[1]),final_cloz.iloc[:,:-1].columns,\
           rotation = 45)
plt.legend()
plt.show()

#final final features
final_mRMRLD = list(final_cloz.iloc[:,:-1].columns)
#25 features that separate clozapine from the others


#make some plots with the combined data
cmap1 = sns.color_palette("tab20", len(uniqueDrugs))
cmap2 = sns.color_palette('Set2', len(uniqueConcs))
for item in final_mRMRLD:
    swarm.swarms('all', item, featMatAll, directoryA, '.tif', cmap1 )
    plt.close()

mrFeatMatFinal3 = pd.concat([mrFeatMatFinal[final_mRMRLD], mrFeatMatFinal['drug'], \
                             mrFeatMatFinal['concentration']], axis= 1)

#make lut for drug colors
lut = dict(zip(uniqueDrugs, cmap1))
lut2 = dict(zip(uniqueConcs, cmap2))
row_colors = mrFeatMatFinal3['drug'].map(lut)#map onto the feature Matrix
row_colors2 = mrFeatMatFinal3['concentration'].map(lut2)

#make clustergram
cg=sns.clustermap(mrFeatMatFinal3.iloc[:,:-2], metric  = 'euclidean', cmap = 'inferno', row_colors = row_colors)
#plt.setp(cg.ax_heatmap.yaxis.set_ticks(order, minor = True))
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels(featMatAll[final_cloz.columns].iloc[cg.dendrogram_row.reordered_ind, -1]))
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels(featMatAll[final_cloz.columns].iloc[cg.dendrogram_row.reordered_ind, -1]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8)
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) 
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'Cloz_mrmrLD.tif'), \
            dpi =150, bbox_inches = 'tight', pad_inches = 1)

drug_order = list(mrFeatMatFinal.iloc[cg.dendrogram_row.reordered_ind, :]['drug'])
conc_order = list(mrFeatMatFinal.iloc[cg.dendrogram_row.reordered_ind, :]['concentration'])
feat_order =list(mrFeatMatFinal3.iloc[:,cg.dendrogram_col.reordered_ind].columns)

#i†erate as a list
colors = [(v) for k,v in lut.items()]
#make figure of color bar
plt.figure()
ax = plt.imshow([colors])
ax.axes.set_xticklabels(range(0,11,1))
ax.axes.set_xticklabels(lut.keys(), rotation = 90)
ax.axes.set_xticks(np.arange(0,9,1))
ax.axes.xaxis.set_ticks_position('top')
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'drug_colors.png'), bbox_inches='tight',dpi =150)

clus_df= pd.DataFrame(data = np.array(drug_order), columns = ['drug'])
clus_df['concentration'] = conc_order
    
"""so basically with 150 features selected by mRMR get as good classification score
as using full feature set
To do still:
    1. See how small a feature set can reduce to still get 97% training efficacy -can use mRMR output
    as these are ranked
        
    2. QDA - paper says that mRMR does not get rid of collinear features so may not work still
    
    3. Classify Chloropromazine vs Clozapine 10uM to see if can pick out features that Chloropromazine
    modulates and not Clozapine
    or just take chloropromazine out of classification
    """

#%% try QDA now
    
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
del clf

clf = QuadraticDiscriminantAnalysis()
clf.fit(X[:,8:12], y)
score2 = clf.score(X_test[:,12:16],y_test)
print(score2)

#does this perform better than if just randomly selected 5 features?

#qda feats
qda_feats = mr_Feats[:4]
for item in qda_feats:
    swarm.swarms('all_qda', item, featMatAll, directoryA, '.tif', cmap1 )
    plt.close()

# 
