#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:24:07 2018

@author: ibarlow
"""

""" incorporation of egg laying data into the antipsychotics analysis"""

from tkinter import Tk, filedialog
import pandas as pd
import numpy as np
import os
import TierPsyInput as TP

#first find the directory to load the data

print ('Select Data Folder')
root = Tk()
root.withdraw()
root.lift()
root.update()
root.directory = filedialog.askdirectory(title = "Select Masked Videos folder", \
                                             parent = root)

directoryA = root.directory
foldersA = os.listdir(directoryA)


#now load all the egg laying data
eggs = {}
fileDir = {}
for rep in foldersA:
    if rep != '.DS_Store':
        if 'Liquid' in rep:
            continue
        else:
            fileDir[rep] = os.listdir(os.path.join(directoryA, rep))
            eggs[rep] = {}
            for line in fileDir[rep]:
                if line.endswith('.csv'):
                    eggs[rep][line] = pd.read_csv(os.path.join(directoryA, rep, line), usecols= ['frame_number', 'x', 'y'])
                    fileDir[rep] = list(eggs[rep].keys())
                else:
                    continue

#extract information about files
drugA = {}
concA = {}
dateA = {}
uniqueIDA = {}
for rep in fileDir:
    drugA[rep], concA[rep], dateA[rep], uniqueIDA[rep] = TP.extractVars(fileDir[rep])
   
#only really care about the number of eggs laid, but will be fun to plot
        #first figure out number of egg laid
eggs2 = {}
for rep in eggs:
    eggs2[rep] = pd.DataFrame()
    for line in eggs[rep]:
        no_frames = np.unique(eggs[rep][line]['frame_number'])
        temp = pd.Series()
        for frame in no_frames:
            temp[str(frame)] = eggs[rep][line][eggs[rep][line]['frame_number'] == frame].shape[0]
        eggs2[rep] = eggs2[rep].append(temp.to_frame().transpose())
        del temp
    eggs2[rep]['drug'] = drugA [rep].values
    eggs2[rep]['concentration'] = concA[rep].values
    eggs2[rep]['date'] = dateA[rep].values
    eggs2[rep]['uniqueID'] = uniqueIDA[rep].values
    
    eggs2[rep] = eggs2[rep].reset_index(drop=True)
    eggs2[rep] =eggs2[rep].fillna(0) #need to replace the nans with zeros

    eggs2[rep]['total'] = np.diff([eggs2[rep].iloc[:,0], eggs2[rep].iloc[:,3]], axis=0).T
    
#calculate total number of eggs laid
    #save these dataframes for loading into the main workspace with the PCA and other features
eggs_all = pd.DataFrame()
for rep in eggs2:
    eggs_all = eggs_all.append(eggs2[rep])
    
fileid = os.path.join(directoryA[:-12], 'egg_all.csv')
eggs_all.to_csv(fileid, index =False)

#this is now ready to be imported into the feature space and can be reused for doing PCA and tSNE
    
    


    
