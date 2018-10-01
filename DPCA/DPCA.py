#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 13:05:20 2018

@author: Heonjun
"""

############################################################
#################### Import packages
import numpy as np
import sklearn.linear_model as lm
from PCA import pca as decomp
from scipy import stats


def dense(a, data):
     kde= stats.gaussian_kde(data)
     return kde(a)
     

def psi(xx, tau = 0.5):
    dim = xx.shape[0]
    cut = 1.345
    result = np.zeros(dim)
    for i in range(dim):
            x = xx[i]
            if x< -cut :
                u = (tau-1)/2
            elif x>= -cut and x<0:
                u = (1-tau)*x
            elif x>= 0 and x<cut:
                u = tau*x
            else:
                u = tau/2
            result[i] = u
    return(result)
    

""" Example
x = np.array([1,2,12,-31,2])
psi(x)
"""


def comp_psi(x, opt_w, tau_range): ## opt_w : n by K
    n, p  = np.shape(x)  # n, p
    result = np.zeros((n,p))
    for i in range(n):
        for k in range(np.shape(tau_range)[0]):
            z = opt_w[i,k] * psi(x[i,],tau = tau_range[k])
            result[i,] = result[i,] + z
    return(result)


''' Example
tau= [0.1,0.2,0.3]

opt_w_row = np.zeros([10,3])
for i in range(np.shape(opt_w_row)[0]):
    opt_w_row[i,] = [0.1,0.4,0.5]

x = np.random.normal(size = (10,3))

comp_psi(x, opt_w_row, tau)

'''



class comp_Pseudo : 
    def __init__(self, n_comp = None):
        self.n_comp = n_comp
        
    def _initial(self, X):
        if self.n_comp is None:
            self.n_comp = min(X.shape)
        
        n, p = X.shape
        
        pca =decomp.PCA(n_components=self.n_comp, scale = False)
        pca.fit(X)
        
        return(pca.components_[:,:self.n_comp])
        
    def fit(self,X):
        
        
            
        B  = self._initial(X)
        hatX = X @ B @ np.transpose(B) 
        
        tau_range = np.arange(0.05,1,0.05)
        opt_w = np.ones((len(X),len(tau_range)))
        
    
        for i in range(np.shape(opt_w)[0]):
            a = np.zeros(len(tau_range))
            
            for k in range(np.shape(opt_w)[1]):
             
                a[k] = float(dense(np.percentile(np.real(X[i]), tau_range[k]),np.real(X[i])))
                
            opt_w[i] = a
            if max(a) >1 :
                opt_w[i] = a/sum(a)
        
        dd =[]
        ans = []
        
        hat_x =[]
        
        for m in range(2):
            Z = hatX + comp_psi(X-hatX,opt_w = opt_w, tau_range=tau_range)
            BX = X @ B
            B =  np.transpose(np.linalg.lstsq(BX, Z)[0])
            hatX1 = X @  B @  np.transpose(B)
            Z = hatX1 + comp_psi(X-hatX1,opt_w = opt_w, tau_range=tau_range)
            
            BX = X @  B
            B =  np.transpose(np.linalg.lstsq(BX, Z)[0])
            hatX2 =  X  @ B @ np.transpose(B) 
            Z = hatX2 + comp_psi(X-hatX2,opt_w = opt_w, tau_range=tau_range)
            ans.append(B)
            dd.append(np.mean(np.square(hatX1-hatX2)))
         
            hat_x.append(hatX2)
            print(hatX2)

        #result = ans[np.argmin(dd)], ans[]
        A = hat_x[np.argmin(dd)]
        E = X-A
        result = A,E
        return(result)
        

