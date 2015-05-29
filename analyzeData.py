#!/usr/bin/python
"""
    Show plots for understanding the data
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.optimize import fmin_cg
import random

from math import sqrt, ceil

features = ['DER_mass_MMC','DER_mass_transverse_met_lep','DER_mass_vis',
'DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet',
'DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt','DER_pt_ratio_lep_tau',
'DER_met_phi_centrality','DER_lep_eta_centrality','PRI_tau_pt',
'PRI_tau_eta','PRI_tau_phi','PRI_lep_pt','PRI_lep_eta','PRI_lep_phi',
'PRI_met','PRI_met_phi','PRI_met_sumet','PRI_jet_num','PRI_jet_leading_pt',
'PRI_jet_leading_eta','PRI_jet_leading_phi','PRI_jet_subleading_pt',
'PRI_jet_subleading_eta','PRI_jet_subleading_phi','PRI_jet_all_pt'
]


def plotDistribution(X, y):
    """
    Plots distribution on each feature to verify
    Gaussian behavior
    """
    slots = 50
    c = 'rgbkmyc'
    m, n = X.shape
    for j in range(n):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        
        F = X[:,j] #only one feature
        F_s = X[y==0,j]
        F_b = X[y==1,j]
        step = (F.max() - F.min()) / slots
        x = np.linspace(F.min(), F.max(), slots)
        #separate signal and background
        s = np.zeros(slots)
        b = np.zeros(slots)
        i = 0
        #create frequency array
        for x1 in x:
            s[i] = (F_s[x1 <= F_s] < x1 + step).sum()
            b[i] = (F_b[x1 <= F_b] < x1 + step).sum()
            i += 1
        #normalize
        s /= m
        b /= m
        plt.title(features[j])
        s = plt.scatter( x, s, s=15, c='r', marker='o', edgecolors='none')
        s = plt.scatter( x, b, s=15, c='b', marker='o', edgecolors='none')
    #plt.show()    
        plt.savefig('distributions%s.png'%features[j])


def convertLabel(l):
    if l=='s':
        return 0
    return 1

def main():    
    bits = 28
                              
    #read the train set
    data = np.loadtxt("data/training.csv", delimiter=',', skiprows=1,
            converters={32: convertLabel})
    #trim
    #data = data[:100,:]
    print "Shape of the Train set", data.shape
    ids = data[:,0] #first column is id
    X_tr = data[:,1:31] #30 features
    w_tr = data[:,31] #weight
    y_tr = data[:,32] #labels (signal or background)
    #print ids
    #print X_tr
    #print w_tr
    #print y_tr
    #how many of each
    plotDistribution(X_tr, y_tr)
    
if __name__ == '__main__':
    main()
