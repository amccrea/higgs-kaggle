"""
    For tunning different parameters about logistic regression
"""

import numpy as np
from utils import *
from loadData import loadData
from LogisticRegression import LogisticRegression

def tuneThreshold():
    """
        Explore different values of threshold to see which one fits best
    """
    thresholds = np.linspace(0.1,0.9, 10)
    
    bestAcc = 0.0
    bestModel = None
    X_tr, y_tr, w_tr = loadData()
    m, n = X_tr.shape
    for th in thresholds:
        model = LogisticRegression(features=['PRI_tau_eta',
                                            'PRI_lep_eta',
                                            'DER_deltar_tau_lep',
                                            'PRI_met_sumet',
                                            'DER_mass_transverse_met_lep'],
                                    threshold=th)
        model.train(X_tr, y_tr, w_tr)
        p, r = model.predict(X_tr)
        #calculate some accuracy on the same train set
        acc =  100.0*(p.flatten() == y_tr.flatten()).sum()/m
        print "%s %s%%"%(th, acc)
        if acc > bestAcc:
            bestAcc = acc
            bestModel = model
    
    #save the best model
    bestModel.save('data/logisticRegression%.2f.txt'%acc)
    
    
def main():
    tuneThreshold()
    
if __name__ == '__main__':
    main()

