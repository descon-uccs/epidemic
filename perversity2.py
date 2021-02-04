# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 15:36:49 2020

@author: Philip Brown
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw

def Rinf(xi,R0=2,eta=.001,scalar=100) :
    w = np.real(lambertw(-R0*(1-eta)*xi*np.exp(-R0*xi)))
    return xi+w/R0

def deriv(func,xi,window=0.000001,R0=2,eta=0.001,scalar=100) :
    return (func(xi+window,R0=R0,eta=eta,scalar=scalar) -
             func(xi,R0=R0,eta=eta,scalar=scalar))/window
            
def Rderiv(xi,R0=2,eta=0.001,scalar=100,window=0.0000001) :
    return deriv(Rinf,xi,window,R0=R0,eta=eta,scalar=scalar)

def Rprob(xi,R0=2,eta=0.001,scalar=100) :
    return Rinf(xi,R0,eta)/xi

P = Rprob

def Pdown(xi,R0=2,eta=0.001,scalar=100) :
    return 2*(1-P(xi,R0,eta,scalar))

## numerical derivatives:

def Pderiv(xi,R0=2,eta=0.001,scalar=100,window=0.0000001) :
    return deriv(P,xi,window,R0=R0,eta=eta,scalar=scalar)

def Pdderiv(xi,R0=2,eta=0.001,scalar=100,window=0.000001) :
    return deriv(Pderiv,xi,window,R0=R0,eta=eta,scalar=scalar)
    
    

## derivatives from implicit stuff in onenote:
    
def PderivImp(xi,R0=2,eta=0.001,scalar=100) :
    # based on Full Bore on Derivatives Page; red star
    RinfinityHere = Rinf(xi,R0,eta,scalar)
    num = (1-eta)*R0*P(xi,R0,eta,scalar)
    denom = (np.exp(R0*RinfinityHere)-(1-eta)*R0*xi)
    return num/denom

def RderivImp(xi,R0=2,eta=0.001,scalar=100) :
    # based on Full Bore on Derivatives Page; double red star
    RinfinityHere = Rinf(xi,R0,eta,scalar)
    num = 1-(1-eta)*np.exp(-R0*RinfinityHere)
    denom = 1-(1-eta)*xi*R0*np.exp(-R0*RinfinityHere)
    return num/denom
    
def PdderivImp(xi,R0=2,eta=0.001,scalar=100) :
    # based on Full Bore on Derivatives Page; triple red star
    RinfinityHere = Rinf(xi,R0,eta,scalar)
    RinfPrimeHere = RderivImp(xi,R0,eta,scalar)
    Phere = P(xi,R0,eta,scalar)
    num = (1-eta)*R0*R0*Phere
    denom = (np.exp(R0*RinfinityHere)-(1-eta)*R0*xi)**2
    parens = 2*(1-eta) - RinfPrimeHere*np.exp(R0*RinfinityHere)
    return num/denom*parens

def PdderivImpEarly(xi,R0=2,eta=0.001,scalar=100) :
    # based on Full Bore on Derivatives Page; circled in RAINBOW
    RinfinityHere = Rinf(xi,R0,eta,scalar)
    RinfPrimeHere = RderivImp(xi,R0,eta,scalar)
    Phere = P(xi,R0,eta,scalar)
    PPrimehere = PderivImp(xi,R0,eta,scalar)
    exp = np.exp(R0*RinfinityHere)
    
    num1 = (1-eta)*R0*PPrimehere
    denom1 = exp - (1-eta)*R0*xi
    
    num2 = (1-eta)*R0*Phere
    denom2 = denom1**2
    
    parens = R0*RinfPrimeHere*exp-(1-eta)*R0
    
    return num1/denom1 - num2/denom2*parens
    
def RinfPrimeExp(xi,R0=2,eta=0.001,scalar=100) :
    RinfinityHere = Rinf(xi,R0,eta,scalar)
    RinfPrimeHere = RderivImp(xi,R0,eta,scalar)
    return RinfPrimeHere*np.exp(R0*RinfinityHere)

def S(xi,R0=2,eta=0.001,scalar=100) :
    return 2*(1-eta)-RinfPrimeExp(xi,R0,eta,scalar)

#def PdderivImp()
    
def f(xi,R0=2,eta=0.001,scalar=100) :
    return 1/xi/scalar;

def cost(xi,R0=2,eta=.001,scalar=100) :
    # this is Js in paper
    return Rprob(xi,R0,eta,scalar) + f(xi,R0,eta,scalar)
            
def costderiv(xi,R0=2,eta=0.001,scalar=100,window=0.0000001) :
    return deriv(cost,xi,window,R0=R0,eta=eta,scalar=scalar)

def Cost(numInUse,R0=2,eta=.001,scalar=100) :
    # assuming all used locations have equal xi
    # return numInUse * (1/scalar + Rinf(1/numInUse,R0,eta,scalar))
    return cost(1/numInUse,R0,eta,scalar) # some bozo thought this was it

def findOptimal(R0=2,eta=0.001,scalar=100) :
    argmin = 1
    min_cost = Cost(1,R0,eta,scalar)
    i = 2
    while True :
        new_cost = Cost(i,R0,eta,scalar)
        if new_cost >= min_cost :
            if i > R0*2 :
                return argmin, min_cost
        else :
            min_cost = new_cost
            argmin = i
        i += 1
        
def isESS(numLoc,R0=2,eta=0.001,scalar=100) :
    # is the allocation with 1/numLoc people at each of numLoc locations 
    # an ESS? To see, inspect the slope of the location cost function.
    # if decreasing at this allocation, this is not an ESS.
    return cost(1/numLoc,R0,eta,scalar) < cost(1/numLoc*1.0001,R0,eta,scalar)
         

def findESS(R0=2,eta=0.001,scalar=100) :
    # Algorithm: note that Optimal is a candidate selfish NE.
    # asdf
    return 1
        

def derivRinf(xi,R0=2,eta=0.001,scalar=100) :
    # note: this function may not be correct at all
    lambertInput = -R0*(1-eta)*xi*np.exp(-R0*xi)
    w = np.real(lambertw(lambertInput))
    expTerm = (1-eta)*(1-R0*xi)*np.exp(-R0*xi)*w
    return 1-expTerm+w/(lambertInput*R0)/(1+w)
    

def plotRinf(R0,eta=0.001,scalar=100) :
    xx = np.arange(0.0,1.01,0.01)
    RRinf = np.empty_like(xx)
    for i in range(len(xx)) :
        RRinf[i] = Rinf(xx[i],R0,eta)
    plt.plot(xx,RRinf)
    
def plotRprob(R0,eta=0.001) :
    xx = np.arange(0.01,1.01,0.01)
    Rprob = np.empty_like(xx)
    for i in range(len(xx)) :
        Rprob[i] = Rprob(xx[i],R0,eta)
    plt.plot(xx,Rprob)
    
    
def plotit(func,R0=2,eta=0.001,scalar=100,numPoints=100,fignum=1) :
    xx = np.linspace(0.001,1,numPoints)
    toplot = np.empty_like(xx)
    for i in range(len(xx)) :
        toplot[i] = func(xx[i],R0,eta,scalar)
    plt.figure(fignum)
    plt.plot(xx,toplot)
#    plt.ylim([0,1])
    
def plotCost(R0=2,eta=0.001,scalar=100,numPoints=10,fignum=13,integer=True) :
    if integer:
        numnum = np.arange(1.,numPoints+1)
    else:
        numnum = np.linspace(1,10,100)
    toplot = np.empty_like(numnum)
    for i in range(len(numnum)) :
        toplot[i] = Cost(numnum[i],R0,eta,scalar)
    plt.figure(fignum)
    plt.plot(numnum,toplot)
    return numnum, toplot
    
    
## more random junk:

# root computation to characterize inflection points of p:
# an inflection point of p is where the graph of p(x) crosses plusroot or minusroot

def plusroot(xi,R0=2,eta=.001,scalar=100) :
    return 1-(3+np.sqrt(9-8*xi*R0))/(4*xi*R0)

def minusroot(xi,R0=2,eta=.001,scalar=100) :
    return 1-(3-np.sqrt(9-8*xi*R0))/(4*xi*R0)

def xfunc(p,R0=2,eta=0.001,scalar=100) :
    return (2-3*p)/(2*R0*(1-p)*(1-p))

## root computation to characterize inflection points of Rinf:
# an inflection point of Rinf is where the graph of Rinf(x) crosses Rplusroot or Rminusroot
    
def Rplusroot(xi,R0=2,eta=.001,scalar=100) :
    squareroot = np.sqrt(R0*R0*pow(xi,4)+2*R0*R0*pow(xi,3)+ R0*(R0-8)*xi*xi-4*R0*xi+4)
    return 1/xi/R0 + (1-xi)/2 + squareroot

def Rminusroot(xi,R0=2,eta=.001,scalar=100) :
    squareroot = np.sqrt(R0*R0*pow(xi,4)+2*R0*R0*pow(xi,3)+ R0*(R0-8)*xi*xi-4*R0*xi+4)
    return 1/xi/R0 + (1-xi)/2 - squareroot
    
