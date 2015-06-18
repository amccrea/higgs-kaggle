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
import random,  itertools
from loadData import loadData

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


def plotDistribution(X, y, w, countWeights=False):
    """
    Plots distribution on each feature to verify
    Gaussian behavior
    """
    slots = 50
    c = 'rgbkmyc'
    m, n = X.shape
    for j in range(n):
        F = X[:,j] #only one feature
        #filter out unknowns
        F = F[F != -999.0].flatten()
        wf = w[F != -999.0].flatten()
        yf = y[F != -999.0].flatten()
        
        #seperate signal and background
        F_s = F[yf==1]
        w_s = wf[yf==1]
        F_b = F[yf==0]
        w_b = wf[yf==0]
        step = (F.max() - F.min()) / slots
        x = np.linspace(F.min(), F.max(), slots)
        s = np.zeros(slots)
        b = np.zeros(slots)
        i = 0
        #create frequency arrays
        for x1 in x:
            #filter feature value by range
            i_s = F_s[x1 <= F_s] < x1 + step
            i_b = F_b[x1 <= F_b] < x1 + step
            #sum weights of events in the range
            if countWeights:
                s[i] = w_s[i_s].sum()
                b[i] = w_b[i_b].sum()
            #if not, only count the events in the range
            else:
                s[i] = i_s.sum()
                b[i] = i_b.sum()
            i += 1
        #normalize over the full weight sum
        if countWeights:
            s /= w_s.sum()
            b /= w_b.sum()
        #otherwise normalize over the # of events on each
        else:
            s /= F_s.size
            b /= F_b.size
        plt.title(features[j])
        #not plot zeros
        s = plt.scatter( x[s!=0.0], s[s!=0.0], s=15, c='r', marker='o', edgecolors='none')
        b = plt.scatter( x[b!=0.0], b[b!=0.0], s=15, c='b', marker='o', edgecolors='none')
    #plt.show()    
        plt.savefig('distributions%s.png'%features[j])
        plt.close()

def plot2DFeatures(X, y, w, f1, f2, th=None):
    """
    Plots a 2D f1 vs f2 to see if we can
    outline a boundary
    """
    #filter weight threshold
    print X.shape, w.shape, y.shape, f1, f2

    if th is not None:
        ind = w >= th
        X = X[ind,:]
        y = y[ind]
        w = w[ind]
    print X.shape, w.shape, y.shape, f1, f2
    #filter out uknowns
    ind = np.logical_and(X[:,f1] != -999.0, X[:,f2] != -999.0)
    X = X[ind,:]
    y = y[ind].flatten()
    w = w[ind].flatten()
    
    #and background
    F1_b = X[y==0, f1]
    F2_b = X[y==0, f2]
    plt.scatter( F1_b, F2_b, s=1, c='b', marker='o', edgecolors='none')

    #signal
    F1_s = X[y==1, f1]
    F2_s = X[y==1, f2]
    #filter out uknowns
    plt.scatter( F1_s, F2_s, s=1, c='r', marker='o', edgecolors='none')

    plt.xlabel(features[f1])
    plt.ylabel(features[f2])
    plt.savefig('%s_vs_%s.png'%(features[f1], features[f2]))
    plt.close()
    
def convertLabel(l):
    if l=='s':
        return 1
    return 0

def main():    
    bits = 28
    
    #load data
    X_tr, y_tr, w_tr = loadData()                          
    print "spaes1", X_tr.shape
    plotDistribution(X_tr, y_tr, w_tr)
    
    #select some features for plotting
    sel_features = [
    features.index('PRI_tau_eta'),
    features.index('PRI_lep_eta'),
    features.index('DER_deltar_tau_lep'),
    features.index('PRI_met_sumet'),
    features.index('DER_mass_transverse_met_lep')]
    print "spaes2", X_tr.shape
    #and make all 2D combinations possible
    for f1, f2 in itertools.combinations(sel_features, 2):
        plot2DFeatures(X_tr, y_tr, w_tr, f1, f2, th=0.0)
    
if __name__ == '__main__':
    main()
