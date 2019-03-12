#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:41:15 2018

@author: ibarlow
"""

""" this script is a test for how to do cross validation. mrFeatMatfinal comes
from running the mRMR classifier script so run this before trying to use any 
of the stuff here as otherwise will get errors"""


from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

#make a conditions dataframe
conds = pd.concat([mrFeatMatFinal['drug'], mrFeatMatFinal['concentration']], axis =1)

#make this condition so just clozapine at 10um vs all the others - 2 conditions
    #still include original drug
conds2 = conds.copy()
conds2['class'] = conds['drug']
for line in conds2.iterrows():
    if line[1]['drug'] == 'Clozapine' and line[1]['concentration'] == 10:
        line[1]['drug'] = 'Clozapine10'
        line[1]['class'] = 'Clozapine10'
    else:
        line[1] ['drug'] =line[1]['drug']
        line[1] ['class'] = 'Other'

#make into arrays for classification
X = np.array(mrFeatMatFinal.iloc[:,:-3])
y = np.array(conds2['class'])

#scale the data
scaler = StandardScaler()
X2 = scaler.fit_transform(X)

#for splitting the data and crossvalidation
    #shuffle 10-fold cross validation
sss = StratifiedShuffleSplit(n_splits = 10, train_size=0.5)

#set liniar discriminant classifier
clf = LinearDiscriminantAnalysis(n_components = None, priors = None, \
                                 shrinkage = 'auto', solver = 'eigen')

#initialise lists for storing the multiple rounds of cross validation
score = []
class_scaling = []
class_means = []
X3=[]
coefs=[]
#start cross-validation
for train_index, test_index in sss.split(X2, y):
    #split the testing and training data
    X_train, X_test = X2[train_index], X2[test_index]
    y_train, y_test = y[train_index], y[test_index]
   
    #then do LDA using these training/testing sets
        #store the transformed scaling for each set
    X3.append(clf.fit_transform(X_train, y_train))
    
    #store the classification score
    score.append(clf.score(X_test, y_test))
   
    #weightings for each feature for each run
       #clf.classes_ = 'Clozapine10', 'other'
    coefs.append(clf.coef_[0])
    
    #scaling matrix - from PCA of feature space
    class_scaling.append(clf.scalings_)
    
    #mean (raw) for each class
    class_means.append(clf.means_)

    del X_train, X_test, y_train, y_test

classes = clf.classes_
del clf

""" now want to pick out the top features that contribute to each round of CV-LDA need 
to pick out the top and bottom ranking features"""

#rank the coef scores for each round
temp = []
for i in coefs:
    temp.append(np.array(stats.rankdata(i)))

#make into a dataframe
feat_scores = pd.DataFrame(data = np.array(temp), columns = mr_Feats2)
del temp

#now find out which features are in the top and bottom 10%, and then only take forward \
#those that are in the top/bottom 10% in 50% of the CVs
bottom = list(feat_scores.columns[np.sum(feat_scores<=15)>=5]) #bottom 10%
top = list(feat_scores.columns[np.sum(feat_scores>=135)>=5]) #top 10%

#combine
combi = top + bottom 
#22 features

#plot these as swarms:
import feature_swarms as swarm
for item in combi:
    swarm.swarms('all', item, featMatAll, directoryA, '.tif', cmap1 )
    plt.close()


#make a dataframe of this final featurespace
final_feat = pd.concat([mrFeatMatFinal[combi], mrFeatMatFinal.iloc[:,-3:]], axis=1)
final_feat['class'] = conds2['drug']

#make a clustergram
    #1. make lut for drug colors
#make a colormap and lookup table
cmap1 = sns.color_palette("tab20", np.unique(final_feat['class']).shape[0])
lut = dict(zip(np.unique(final_feat['class']), cmap1))

#add in row colors to the dataframe
row_colors = final_feat['class'].map(lut)#map onto the feature Matrix

#make clustergram
cg=sns.clustermap(final_feat.iloc[:,:-4], metric  = 'euclidean', cmap = 'inferno', \
                  row_colors = row_colors)
#plt.setp(cg.ax_heatmap.yaxis.set_ticks(order, minor = True))
plt.setp(cg.ax_heatmap.yaxis.set_ticklabels\
         (final_feat['class'][cg.dendrogram_row.reordered_ind]))
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize = 8)
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation = 90, fontsize = 10) 
col = cg.ax_col_dendrogram.get_position()
cg.ax_col_dendrogram.set_position([col.x0, col.y0, col.width*1, col.height*1])
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'Cloz_mrmrLD2.tif'), \
            dpi =150, bbox_inches = 'tight', pad_inches = 1)

#make list of the final order of the drugs
drug_order = list(final_feat['class'][cg.dendrogram_row.reordered_ind])
conc_order = list(final_feat['ccncentration'][cg.dendrogram_row.reordered_ind])
feat_order =list(final_feat.iloc[:,:-4].columns[cg.dendrogram_col.reordered_ind])

#make a figure of the colorbar
colors = [(v) for k,v in lut.items()]

#make figure of color bar
plt.figure()
ax = plt.imshow([colors])
ax.axes.set_xticklabels(range(0,11,1))
ax.axes.set_xticklabels(lut.keys(), rotation = 90)
ax.axes.set_xticks(np.arange(0,10,1))
ax.axes.xaxis.set_ticks_position('top')
plt.savefig(os.path.join(directoryA[0:-7], 'Figures', 'drug_colors.png'),\
            bbox_inches='tight',dpi =150)

