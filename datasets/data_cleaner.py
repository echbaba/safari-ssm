#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 17:38:15 2025

@author: hossein
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

def resample(x, L2):
    L=len(x)
    if np.iscomplexobj(x):
        Z= np.zeros(3*L) * (1j)
    else:
        Z= np.zeros(3*L)
    Z[0:L]=np.flip(x)
    Z[L:2*L]=x
    Z[2*L:3*L]=np.flip(x)
    Y= scipy.signal.resample(Z, 3*L2)
    return Y[L2:2*L2]


################################################
#####################  SP500  ##################
################################################
data_path='sp500/'

df_all= pd.read_csv(data_path+'year1.csv')
#print("scanned year", 1, ".scv")

for i in range(9):
    df= pd.read_csv(data_path+'year'+str(i+2)+'.csv')
    #print("scanned year", i+2, ".scv")
    df_all = pd.concat([df, df_all], ignore_index=True)


##creating time series instances of specific length and overlap, then resample it to a given length
L_instance=500
L_output=4000
overlap=100
start_ind= np.arange(0, len(df_all)-L_instance, overlap)
x_dump= np.zeros((4*len(start_ind), L_output))
for i,start in enumerate(start_ind):
    print(start,":", start+L_instance)
    x1 = df_all["Open"].iloc[start:start+L_instance].str.replace(",", "").astype(float).to_numpy()
    x2 = df_all["High"].iloc[start:start+L_instance].str.replace(",", "").astype(float).to_numpy()
    x3 = df_all["Low"].iloc[start:start+L_instance].str.replace(",", "").astype(float).to_numpy()
    x4 = df_all["Close"].iloc[start:start+L_instance].str.replace(",", "").astype(float).to_numpy()

    x_dump[4*i, :] = resample(x1,L_output)
    x_dump[4*i+1, :] =resample(x2,L_output)
    x_dump[4*i+2, :] = resample(x3,L_output)
    x_dump[4*i+3, :] = resample(x4,L_output)

np.savetxt(data_path+"procesed.csv", x_dump, delimiter=",")
print("saved the aggregated version of SP500 dataset")

################################################
##################### More ##################
################################################
"""
    There is more data here, but I will only use this for now. Once we have experiments ready, we can come back here and ask for more data to put into the apper
"""