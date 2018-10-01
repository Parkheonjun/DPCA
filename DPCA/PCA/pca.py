#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 07:05:32 2018

@author: Heonjun
"""
import numpy as np
from numpy import linalg as LA


class PCA():
    def __init__(self, n_components = None, scale = True):
        self.n_components = n_components
        self.scale = scale
        
    def fit(self, X):
        if self.n_components is None:
            self.n_components = min(X.shape)
        
        if self.scale is True:
            V =  np.corrcoef(X,rowvar=False)
        else:
            V = np.cov(X,rowvar=False)
        
        self.variance_, self.components_ = LA.eig(V)
      

