#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 09:58:58 2021

@author: kebaiera
"""


import numpy as np
import numpy.random as npr 
import matplotlib.pyplot as plt
from scipy.stats import norm


# Simulation of Brownian path cumsum
 
T=1 # One year
n=100 # number of points

d=5 # d paths of d BM

G=np.sqrt(T/n)*npr.normal(size=(5,n))



B=np.concatenate((np.zeros((5,1)),np.cumsum(G,axis=1)),axis=1) #horizontal cumsum for each row

print(B)

x=np.linspace(0,1,n+1)

plt.plot(x,np.transpose(B))

# Second method

Gamma=np.zeros((n,n))

for i in range(n):
    for j in range(n):
        Gamma[i,j]=np.minimum(i+1,j+1)
        
Gamma=(T/n)*Gamma

A=np.linalg.cholesky(Gamma)


G=npr.normal(size=(n,1))

X=np.dot(A,G)

print(X)      

plt.plot(X) 
    

# Use Multivariate function of Python (More optimized )

Gamma=np.zeros((n,n))

for i in range(n):
    for j in range(n):
        Gamma[i,j]=np.minimum(i+1,j+1)
        
Gamma=(T/n)*Gamma

Brownian_Path=np.random.multivariate_normal(np.zeros(n),Gamma,size=d)

plt.plot(np.transpose(Brownian_Path))

# Pricing Asian Options Under B-S model


def Asian_call_MC_BS(r,S0,sigma,T,K,m,n):
    
    delta=float(T/n)

    G=npr.normal(0,1,size=(m,n))

    #Log returns
    LR=(r-0.5*sigma**2)*delta+np.sqrt(delta)*sigma*G
    # concatenate with log(S0)
    LR=np.concatenate((np.log(S0)*np.ones((m,1)),LR),axis=1)
    # cumsum horizontally (axis=1)
    LR=np.cumsum(LR,axis=1)
    # take the expo Spath matrix
    Spaths=np.exp(LR)
    #print(Spaths)
    
    ## Riemann approximation
    #remove final time component
    Spaths=Spaths[:,0:len(Spaths[0,:])-1]
    #print(Spaths)
    
    #take the average over each row
    Sbar=np.mean(Spaths,axis=1)
    #print(Sbar)
    payoff=np.exp(-r*T)*np.maximum(Sbar-K,0) #call function

    Asian_MC_price=np.mean(payoff)

    # 95% C.I

    Stdev=np.std(payoff) # standard deviation estimator

    error=1.96*Stdev/np.sqrt(m)

    CI_up=Asian_MC_price + error
    CI_down=Asian_MC_price -error
    
    return Asian_MC_price,CI_up,CI_down,error




[Asian_MC_price,CI_up,CI_down,error]=Asian_call_MC_BS(0.04,100,0.2,1,100,1000000,100)    
print('*******MC_Price*****')
print(Asian_MC_price,CI_up,CI_down,error)




