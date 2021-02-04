# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:25:11 2020

@author: Philip Brown
"""

import numpy as np
import matplotlib.pyplot as plt

class Epidemic :
    def __init__(self,alphas,KK,delta,T,BB,p0,a0,gamma=0.01) :
        # alphas = [a12, a21]
        # KK = [K1,K2]
        # BB = [Bx,By]
        # p0 is initial infection state [p1,p2] - each p_i must be in [0,1]
        # a0 is initial behavior state [axx,axy,ayx,ayy] - must be a valid probability distribution
        self.A = np.array([[0,alphas[0]],
                           [alphas[1],0]])
        self.K1 = KK[0]
        self.K2 = KK[1]
        self.D = np.identity(2)*delta
        self.Bx = BB[0]
        self.By = BB[1]
        self.p0 = np.array(p0).reshape([2,1])
        self.avgp = [self.p0]
        self.pHist = [self.p0]
        self.p = self.p0.copy()
        self.a0 = np.array(a0).reshape([4,1])
        self.a = self.a0.copy()
        self.aHist = [self.a0]
        self.T = T
        self.gamma = gamma
        
        e1T = np.exp(1/T)
        self._baseP = np.zeros([4,4])
        self._baseP[1,3] = e1T/(1+e1T)
        self._baseP[3,1] = 1/(1+e1T)
        self._baseP[3,2] = 1/(1+e1T)
        self._baseP[2,3] = e1T/(1+e1T)
        self._baseP = self._baseP*0.5
        self._baseP[3,3] = 1 - self._baseP[3,2] - self._baseP[3,1]
        
    def getP(self) :
        P = np.zeros([4,4])
        ep1T = np.exp((self.K1+self.K2*self.p[0])/self.T)
        ep2T = np.exp((self.K1+self.K2*self.p[1])/self.T)
        P[0,1] = 1/(1+ep2T)
        P[0,2] = 1/(1+ep1T)
        P[1,0] = ep2T/(1+ep2T)
        P[2,0] = ep1T/(1+ep1T)
        P = P*0.5
        P = P + self._baseP
        for i in range(3) :
            P[i,i] = 1-sum(P[i,:])
        return P
    
    def runDynamics(self,numSteps=100000,samplePaths=False,disableBehavior=False,disableEpidemic=False) :
        for i in range(numSteps) :
            if disableEpidemic :
                nextp = self.p.copy()
            else :
                B = np.zeros([2,2])
                B[0,0] = self.Bx*(self.a[0]+self.a[1]) + self.By*(self.a[2]+self.a[3])
                B[1,1] = self.Bx*(self.a[0]+self.a[2]) + self.By*(self.a[1]+self.a[3])
                nextp = self.p + (np.diagflat(1-self.p)@(self.A.T)@B-self.D)@self.p*self.gamma
            if disableBehavior :
                nexta = self.a.copy()
            else :
                nexta = np.zeros([4,1])
                aprobs = self.getP().T@self.a
                if samplePaths :
                    nextState = np.random.choice([0,1,2,3],p=aprobs.flatten())
                    nexta[nextState] = 1
                else :
                    nexta = aprobs                    
            self.pHist.append(nextp)
            self.aHist.append(nexta)
            self.p = nextp.copy()
            self.a = nexta.copy()
            self.avgp.append(1/(i+1)*(self.p + i*self.avgp[-1]))
        
    def plotInfectionHistory(self) :
        plt.plot([p[0] for p in self.pHist])
        plt.plot([p[1] for p in self.pHist])
        plt.plot([np.mean(p) for p in self.avgp])
        
        
if __name__ == "__main__" :
    plt.figure(1)
    test = Epidemic([1,2],[0.5,1],.1,.1,[.2,1],[.01,0],[0,0,0,1],0.001)
    test.runDynamics(numSteps=200000,samplePaths=False)
    test.plotInfectionHistory()
    
    plt.figure(2)
    test = Epidemic([1,2],[0.5,1],.1,.1,[.2,1],[.01,0],[0,0,0,1],0.001)
    test.runDynamics(numSteps=200000,samplePaths=True)
    test.plotInfectionHistory()