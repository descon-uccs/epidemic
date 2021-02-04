# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:00:29 2020

@author: Philip Brown
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def rhs(rinf,x,betabar,gamma,r0,s0) :
    return 1-s0*np.exp(-betabar*x*x/gamma*(rinf-r0))

def lhs(rinf) :
    return rinf

rr = np.arange(0,1,.001)
lefts = np.empty_like(rr)
rights = np.empty_like(rr)

x = 1
gamma = .1
betabar = .3
r0 = 0.0
s0 = 0.999

for i in range(len(rr)) :
    lefts[i] = lhs(rr[i])
    rights[i] = rhs(rr[i],x,betabar,gamma,r0,s0)
    
plt.clf()
plt.plot(rr,lefts)
plt.plot(rr,rights)