# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:09:45 2015
"""

import numpy as np
import os

def convertLabel(l):
    """
    Label converter function for loading the data
    """
    if l=='s':
        return 1
    return 0

# Debug mode off by default
def loadData(ddir='data/training.csv', debug=False, trim=False):
    """
        Loads the training data
        if trim is an integer, it picks only the first trim samples
    """

    if debug==1:
        print "CWD: ",os.getcwd(),"\n"

    """
        Load the data in a single go (faster than reading it three times)
        Using a column converter for the label value
    """
    data = np.loadtxt(ddir,
                        delimiter=',',
                        skiprows=1,
                        converters={32: convertLabel})
    if debug:
        print "Shape of data", data.shape
    
    if trim:
        data = data[:trim,:]
    
    '''
    X = The 30 features
    '''
    X = data[:,:31]
    '''
    W = Weights of a given sample. Simulators produce weights for each event to
    correct for the mismatch between the natural (prior) probability of the event
    and the instrumental probability applied by the simulator.
    Reshaped into (m x 1) vector
    '''
    w = data[:,31]
    '''
    Y = "s" or "b" classification for signal and background.
    '''
    Y = data[:,32]

    if debug==1:
        print "X:\n%s\n\nY (numeric):\n%s\n\nW:\n%s\n\n" %(X, Y, W)
        print "Shape of X: %s\nShape of Y: %s\nShape of W: %s\n"\
                                            %(X.shape, Y.shape, W.shape)
    return X, Y, W

if __name__ == '__main__':
    loadData(debug=True)
