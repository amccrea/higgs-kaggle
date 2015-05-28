#!/usr/bin/python
"""
    Show plots for understanding the data
"""
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.optimize import fmin_cg
import random

from math import sqrt, ceil

features = ['EventId','DER_mass_MMC','DER_mass_transverse_met_lep',
  'DER_mass_vis','DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet',
  'DER_prodeta_jet_jet','DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt',
  'DER_pt_ratio_lep_tau','DER_met_phi_centrality','DER_lep_eta_centrality',
  'PRI_tau_pt','PRI_tau_eta','PRI_tau_phi','PRI_lep_pt','RI_lep_eta,PRI_lep_phi',
  'PRI_met','PRI_met_phi','PRI_met_sumet','PRI_jet_num','PRI_jet_leading_pt',
  'PRI_jet_leading_eta','PRI_jet_leading_phi','PRI_jet_subleading_pt',
  'PRI_jet_subleading_eta','PRI_jet_subleading_phi','PRI_jet_all_pt','Weight']


def plotDistribution(X):
    """
    Plots distribution on each feature to verify
    Gaussian behavior
    """
    slots = 50
    c = 'rgbkmyc'
    m, n = X.shape
    for j in range(n):
        F = X[:,j] #only one feature
        step = (F.max() - F.min()) / slots
        x = np.linspace(F.min(), F.max(), slots)
        y = np.zeros(slots)
        i = 0
        #create frequency array
        for x1 in x:
            y[i] = (F[x1 <= F] < x1 + step).sum()
            i += 1
        #normalize
        y /= y.sum()
        sub = plt.subplot(5,6,j+1)
        sub.set_title(features[j])
        s = sub.scatter( x, y, s=3, c=c[j%len(c)], marker='o', edgecolors='none')
    #plt.show()    
    plt.savefig('distributions.png')


def convertLabel(l):
    if l=='s':
        return 0
    return 1

def main():    
    bits = 28
                              
    #read the train set
    data = np.loadtxt("training.csv", delimiter=',', skiprows=1,
            converters={32: convertLabel})
    #trim
    data = data[:100,:]
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
    plotDistribution(X_tr)
    
if __name__ == '__main__':
    main()
