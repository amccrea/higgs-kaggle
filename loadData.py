# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:09:45 2015
"""

import numpy as np
import os

# Debug mode off by default
def loadData(ddir='data/training.csv', debug=False, trim=False):
    """
        Loads the training data
        if trim is an integer, it picks only the first trim samples
    """

    if debug==1:
        print "CWD: ",os.getcwd(),"\n"

    '''
    X = The 30 features
    '''
    X = np.genfromtxt(ddir,delimiter=',',
                      skip_header=1,usecols=(range(1,31)))
    if trim:
        X = X[:trim,:]
    '''
    Y = "s" or "b" classification for signal and background.
    Reshaped into (m x 1) vector
    '''
    Y = np.genfromtxt(ddir,delimiter=',',
                      skip_header=1,usecols=(32),dtype="S1")
    if trim:
        Y = Y[:trim]
    #Y = np.reshape(Y,(-1,1))
    #Y_numeric = np.reshape([1 if i=="s" else 0 for i in Y],(-1,1))
    #this is faster
    Y_numeric = np.zeros((Y.shape[0],1))
    Y_numeric[Y == "s"] = 1

    '''
    W = Weights of a given sample. Simulators produce weights for each event to
    correct for the mismatch between the natural (prior) probability of the event
    and the instrumental probability applied by the simulator.
    Reshaped into (m x 1) vector

    '''
    W = np.genfromtxt(ddir,delimiter=',',
                      skip_header=1,usecols=(31))
    W = np.reshape(W,(-1,1))
    if trim:
        W = W[:trim,:]

    if debug==1:
        print "X:\n%s\n\nY (numeric):\n%s\n\nW:\n%s\n\n" %(X, Y_numeric, W)
        print "Shape of X: %s\nShape of Y: %s\nShape of W: %s\n"\
                                            %(X.shape, Y.shape, W.shape)
    return X, Y_numeric, W

if __name__ == '__main__':
    loadData(debug=True)
