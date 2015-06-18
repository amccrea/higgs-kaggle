#!/usr/bin/python
"""
    A fake model for creating a fake prediction
"""

import numpy as np
from math import sqrt, ceil

class FakeModel:
  def __init__(self, **kargs):
    pass
    
  def train(self, X, y, w):
    #we do nothing
    pass
  
  def predict(self, X):
    #create a random prediction
    m, n = X.shape
    y = np.random.random(m)
    l = np.array(range(m), dtype=str)
    for i in range(m):
      l[i] = 's' if y[i] > 0.5 else 'b'
    #create a consistent ranking based on sorting by y
    # higher Y == S
    r = np.argsort(y)
    #shift to 1-ranked
    r += 1
    return l, r
    
