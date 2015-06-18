#!/usr/bin/python
"""
    Utilities and formulas
"""
import numpy as np
import scipy as sp
import random
from math import sqrt, ceil

def initMatrix(r, c, e):
    """
    Creates a matrix [r x c] with random
    values between (-e,e)
    """
    m = np.random.rand(r,c)
    m = -e + m*2*e
    return m


def sigmoid(x):
    """
    logistic function
    """
    return 1/(1+np.exp(-x))

def sigmoidGrad(x):
    """
    Derivative of sigmoid
    """
    g = sigmoid(x)
    return g*(1-g)

def remove999s(x):
    """
    Replace all -999s with zeros.
    """
    for i in np.nditer(x,op_flags=['readwrite']):
        if i == -999:
            i[...] = 0

def normalizeFeatures(X,r999s=True,ntype='minmax'):
    """
    Normalize features (columns)
    either by minmax or zscore
    """

    num_features = np.size(X,1) # num_features = 30

    if r999s==True:
        remove999s(X)

    for i in range(num_features):  # For each column (feature) in X...

        col = X[:,i]

        if ntype=="minmax":
            new_col = (col-min(col))/(max(col)-min(col))
        if ntype=="zscore":
            new_col = (col-mean(col))/std(col)
        X[:,i] = new_col
