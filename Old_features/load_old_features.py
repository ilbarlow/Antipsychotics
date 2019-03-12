#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:44:45 2018

@author: ibarlow
"""

#temp script for doing clustering on agar data

from scipy import spatial
import Dcluster as dcl
from tkinter import Tk, filedialog

print ('Select Dat file')
root = Tk()
root.withdraw()
root.lift()
root.update()
root.filename= filedialog.askopenfilename(title = "Select dat file", \
                                             parent = root)


fileid = root.filename

(dist1, xxdist1, ND1 ,N1) = dcl.readfile(file = fileid, sep = '\t')
(Y1, S1) = dcl.mds(dist1)
(rho1, delta1, ordrho1,dc1, nneigh1) = dcl.rhodelta(dist1, xxdist1, ND1, N1)