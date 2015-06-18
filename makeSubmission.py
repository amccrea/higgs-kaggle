#!/usr/bin/python
"""
    Creates a submission file
"""
import numpy as np
from FakeModel import FakeModel
from LogisticRegression import LogisticRegression, transformY


def save(ids, p, r):
    ids = ids.flatten()
    p = p.flatten()
    r = r.flatten()    
    f = open('submission.csv', 'w')
    #Header
    f.write("EventId,RankOrder,Class\n")
    for i in range(ids.size):
        # Transforms 1 to 's' and 0 to 'b'
        f.write("%s,%s,%s\n"%(ids[i],r[i],
                's' if p[i] == 1 else 'b'))
    f.close()
  
def main():
    #model = FakeModel() #TODO model parameters
    model = LogisticRegression(features=['PRI_tau_eta',
                                        'PRI_lep_eta',
                                        'DER_deltar_tau_lep',
                                        'PRI_met_sumet',
                                        'DER_mass_transverse_met_lep'])
    #load some previously saved model parameters                                    
    model.load('data/logisticRegression69.61.txt')
    
    #load test data
    data = np.loadtxt("data/test.csv", delimiter=',', skiprows=1)
    ids = data[:,0].astype(int) #first column is id
    X = data[:,1:31] #30 features
    
    #make predictions (ranking and label)
    r, p = model.predict(X)
    
    #save to file
    save(ids, r, p)
  
if __name__ == '__main__':
    main()
