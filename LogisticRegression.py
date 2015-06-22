#!/usr/bin/python
"""
    A quick logistic regression model
"""
import numpy as np
from utils import *
import scipy as sp
from scipy.optimize import fmin_cg
from loadData import loadData

feat_names = ['DER_mass_MMC','DER_mass_transverse_met_lep','DER_mass_vis',
'DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet',
'DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt','DER_pt_ratio_lep_tau',
'DER_met_phi_centrality','DER_lep_eta_centrality','PRI_tau_pt',
'PRI_tau_eta','PRI_tau_phi','PRI_lep_pt','PRI_lep_eta','PRI_lep_phi',
'PRI_met','PRI_met_phi','PRI_met_sumet','PRI_jet_num','PRI_jet_leading_pt',
'PRI_jet_leading_eta','PRI_jet_leading_phi','PRI_jet_subleading_pt',
'PRI_jet_subleading_eta','PRI_jet_subleading_phi','PRI_jet_all_pt'
]

class LogisticRegression:
    def __init__(self, **kargs):
        print kargs
        #get the position of features
        self.features = [feat_names.index(f) for f in kargs['features']]
        #regularization factor
        self.lmbd = kargs['lmbd'] if 'lmbd' in kargs else 1.0
        #initialization range
        self.epsilon = kargs['epsilon'] if 'epsilon' in kargs else 0.5
        #initialization of maxiter
        self.maxiter = kargs['maxiter'] if 'maxiter' in kargs else 500
        #threshold for deciding which ones are s
        self.threshold = kargs['threshold'] if 'threshold' in kargs else 0.5
        
        self.__buildFeatures__(kargs)
    
    def __buildFeatures__(self, kargs):
        """
        TODO composes polinomial features
        """
        pass
    
    def train(self, X, y, w):
        """
        Trains based on data and saves it
        Assumes: y is a {0,1} matrix wich a column per class
        """
        #filter only selected features and normalize
        X = X[:,self.features]
        normalizeFeatures(X, r999s=False)
        #TODO we need to store the normalization parameters to apply before predicting.
        
        m, self.n = X.shape
        #number of classes
        self.k = y.shape[1] if len(y.shape) == 2 else 1
        #add column of ones
        X = np.insert(X, 0, np.ones(m), axis=1)
        self.n += 1
        
        #TODO calculate composed features
        
        #initial theta one row per feature, one col per class
        ThetaIni = -self.epsilon + 2.0*self.epsilon \
                    * np.random.random((self.n,self.k))
        self.data = None
        self.J = None
        self.grad = None
        
        #optimize
        ThOpt = fmin_cg(self.cost, ThetaIni, fprime = self.grad,
                        args=(X, y),
                        maxiter= self.maxiter, disp=True)
        #cost and grad functions will save Theta Opt, so no need to
        
    
    def cost(self, data, X, y):
        """
        Cost of the given model
        """
        #last calculated cost
        if self.J != None and data is self.data:
            #print "cache cost"
            pass
        else:
            #print "cost"
            self.data = data
            m, n = X.shape
            m, k = y.shape
            #reshape to get theta
            self.Theta = np.reshape(data, (n, k))
            self.__costAndGrad__(X, y)
        return self.J
    
    def grad(self, data, X, y):
        """
        Gradient of the given model
        """
        #last calculated gradient
        if self.grad != None and data is self.data:
            #print "cache grad"
            pass
        else:
            #print "grad"
            self.data = data
            m, n = X.shape
            m, k = y.shape
            #reshape to get theta
            self.Theta = np.reshape(data, (n, k))
            self.__costAndGrad__(X, y)
            #flatten gradient
            self.grad = self.grad.flatten()
        return self.grad
    
    def __costAndGrad__(self, X, y):
        """
        Calculate cost and gradient of the model in one go
        """
        m, n = X.shape
        m, k = y.shape
        
        h = sigmoid(np.dot(X, self.Theta))
        self.J = 1.0/m * np.sum( -y*np.log(h) - (1-y)*np.log(1-h) )
        #regularization term
        self.J += self.lmbd/ 2.0/ m * np.sum( self.Theta[1:]*self.Theta[1:] )
        
        #gradient, one row per feature, one column per class
        #(same dimension than Theta)
        self.grad = 1.0/m * np.dot(X.T, (h - y))
        #regularization term (all but first feature - row)
        self.grad[1:,:] += self.lmbd / m * self.Theta[1:,:]
        
        #print self.J
      
    def predict(self, X):
        """
        Predicts and returns an array of labels and an array of ranking
        """
        #classes
        classes = np.array(['s','b'])
        
        #filter only selected features and normalize
        X = X[:,self.features]
        normalizeFeatures(X, r999s=False)
        m, n = X.shape
        #TODO we need to store the normalization parameters to apply before predicting.
        
        #add column of ones
        n += 1
        X = np.insert(X, 0, np.ones(m), axis=1)
        #hypothesis result
        h = sigmoid(np.dot(X, self.Theta))
        #prediction result
        p = np.zeros(h.shape)
        #which ones have column 0 over the threshold. (column 1 is redundant)
        p[h > self.threshold] = 1
        
        #calculate a ranking based on sorting of column 0 (column 1 is redundant)
        r = np.argsort(h[:,0]) + 1
        return p, r
        
    def save(self, f):
        """
        Stores the model parameters into a text file.
        Do this after training.
        """
        out = open(f,'w')
        #model dimensions
        out.write('%s,%s\n'%(self.n,self.k))
        #model parameters
        self.data = self.Theta.flatten()
        out.write( ','.join(['%s'%x for x in self.data]) )
        out.write('\n')
        out.close()
    
    def load(self, f):
        """
        Load the stored parameters in a text file
        Do this when want to get a saved model
        """
        inp = open(f,'r')
        #read model dimensions
        self.n, self.k = inp.readline().split(',')
        self.n = int(self.n)
        self.k = int(self.k)
        #read parameters
        self.data = inp.readline().split(',')
        self.data = np.array([float(x) for x in self.data])
        self.Theta = np.reshape(self.data, (self.n, self.k))
        
        inp.close()

def transformY(y, negColumn=False):
    """
        Transform
        ['s', 's', 'b', ...]
        into 
        [1, 1, 0, ...]
        If negColumn is True, it also creats an additional column
        with the negation, transforming this
        ['s', 's', 'b', ...]
        into this:
        [ [1, 0],
          [1, 0],
          [0, 1],
          ...
        ]
    """
    if not negColumn:
        y_tr = np.zeros(y.shape[0])
        y_tr[y == 's'] = 1
    else:
        #compose y_tr for class format
        y_tr = np.zeros((y.shape[0], 2))
        y_tr[y == 's',0] = 1
        y_tr[y != 's',1] = 1
    return y_tr

def testLogisticRegression():
    """
        Make a fake dataset to test that the logistic
        regression is working
    """
    n = 2
    m = 1000
    k = 1
    X = np.random.random((m, n))
    y = np.zeros((m, k))
    #first half will point to the class 1
    #so we center the data around 1.0,2.0
    X[:m/2,:] += np.array((1.0, 2.0))
    y[:m/2] = 1
    
    #second half will point to class 2, so we 
    #move the data to 1.8, 2.6 (so there is some overlapping)
    X[m/2:,:] += np.array((1.8, 2.6))
    #import matplotlib.pyplot as plt
    #plt.scatter(X[y[:,0] == 1, 0], X[y[:,0] == 1, 1], c='r' )
    #plt.scatter(X[y[:,1] == 1, 0], X[y[:,1] == 1, 1], c='b' )
    #plt.show()
    
    model = LogisticRegression(features=['DER_mass_MMC','DER_mass_transverse_met_lep'],
    )
    model.train(X, y, None)
    
    p, r = model.predict(X)
    #print p
    #print r
    #calculate some accuracy on the train set
    acc =  100.0*(p.flatten() ==y.flatten()).sum()/m
    print "%s%%"%acc



def trainWithRealData():
    """
        Test with the real-deal
    """
    X_tr, y_tr, w_tr = loadData()
    m, n = X_tr.shape
    model = LogisticRegression(features=['PRI_tau_eta',
                                        'PRI_lep_eta',
                                        'DER_deltar_tau_lep',
                                        'PRI_met_sumet',
                                        'DER_mass_transverse_met_lep'])
    #tune parameters later.
    model.train(X_tr, y_tr, w_tr)
    p, r = model.predict(X_tr)
    #calculate some accuracy on the same train set
    acc =  100.0*(p.flatten() == y_tr.flatten()).sum()/m
    print "%s%%"%acc
    #save the model
    model.save('data/logisticRegression%.2f.txt'%acc)
    
def main():
    #testLogisticRegression()
    trainWithRealData()
    
if __name__ == '__main__':
    main()



