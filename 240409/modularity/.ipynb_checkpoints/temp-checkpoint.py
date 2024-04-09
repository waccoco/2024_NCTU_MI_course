# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import pandas as pd
import scipy.io

XYZ_Coord = scipy.io.loadmat('data\XYZcoord1.mat')
XYZ_C = np.array(XYZ_Coord ['XYZ1'])
XYZ_C
print(len(XYZ_C))
print(XYZ_C)

import math as mt
import random as rd

#預先定義歐是距離函數
def dist(a,b):
    r2 = (a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2
    r = mt.sqrt(r2)
    return r
#先設一個list 來存放點對點歐式距離的結果與資訊
Euclidean_Edges=[0,0,0]

for i in range(1990,1995):  #從第一點到倒數第二點計算
    p1=XYZ_C[i]
    for j in range(i+1,len(XYZ_C)-1):
        p2=XYZ_C[j]
        source=i
        Target=j
        Weight=dist(p1,p2)
        pp=[source,Target,Weight]
        Euclidean_Edges=np.vstack((Euclidean_Edges,pp))
print(Euclidean_Edges)

