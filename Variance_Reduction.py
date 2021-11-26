#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 12:07:11 2021

@author: kebaiera
"""

import numpy as np   # math operations
import numpy.random as npr # random 
import matplotlib.pyplot as plt # plot

from scipy.stats import norm

def European_call_MC_BS_VR(r,S0,sigma,T,K,Sample_size):
    
    G=npr.normal(0,1,size=(1,Sample_size))
    
    
    S=S0*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*G)
    
    Sa=S0*np.exp((r-sigma**2/2)*T-sigma*np.sqrt(T)*G)
    
    
    
    
    
    payoff=np.exp(-r*T)*np.maximum(S-K,0) #call function
    
    payoffa=0.5*(np.exp(-r*T)*np.maximum(S-K,0)+
                 np.exp(-r*T)*np.maximum(Sa-K,0))

    MC_price=np.mean(payoff)
    MCa_price=np.mean(payoffa)
    
    
    

# 95% C.I

    STDV=np.std(payoff) # standard deviation estimator
    STVA=np.std(payoffa)

    error=1.96*STDV/np.sqrt(Sample_size)
    errora= 1.96*STVA/np.sqrt(Sample_size)
    

    #CI_up=MC_price + error
    #CI_down=MC_price -error
    
    return MC_price,MCa_price,error,errora

[MC_price,MCa_price,error,errora]=European_call_MC_BS_VR(0.05,100,0.2,1,100,1000000
)

print(MC_price,MCa_price,error,errora)



##########Control Variate#########@


def European_call_Parity_Formula_MC_BS(r,S0,sigma,T,K,Sample_size):
    
    G=npr.normal(0,1,size=(1,Sample_size))
    S=S0*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*G)
    
    Call_payoff=np.exp(-r*T)*np.maximum(S-K,0) #call function
    Put_payoff=np.exp(-r*T)*np.maximum(K-S,0)

    MC_Call_price=np.mean(Call_payoff)
    MC_Call_price_parity_Formula=np.mean(Put_payoff)+S0-np.exp(-r*T)*K

# 95% C.I

    Call_STDV=np.std(Call_payoff) # standard deviation estimator
    Put_STDV=np.std(Put_payoff)

    Call_error=1.96*Call_STDV/np.sqrt(Sample_size)
    Call_parity_Formula_error=1.96*Put_STDV/np.sqrt(Sample_size)

    
    
    return MC_Call_price, Call_error, MC_Call_price_parity_Formula, Call_parity_Formula_error


[MC_Call_price, Call_error, MC_Call_price_parity_Formula, Call_parity_Formula_error]=European_call_Parity_Formula_MC_BS(0.05,100,0.2,1,100,1000000)

print('********')

print(MC_Call_price, Call_error, MC_Call_price_parity_Formula, Call_parity_Formula_error)

#### Importance Sampling 1 #############


# task compute numerically P(G>7) which is very small but not 0


def survival_prob(N,mu):
    
    
    G=npr.normal(0,1,size=(1,N))
    
    prob=np.mean((G>7))
    
    payoff=(G+mu>7)*np.exp(-0.5*mu**2-mu*G)
    
    prob_IS=np.mean(payoff)
    
    return prob,prob_IS


[prob,prob_IS]=survival_prob(1000000,7)

print('*******')
print(prob,prob_IS)
    


def European_call_MC_BS_IS(r,S0,sigma,T,K,Sample_size,mu):
    
    G=npr.normal(0,1,size=(1,Sample_size))
    S=S0*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*(G+mu))
    
    payoff=np.exp(-r*T)*np.maximum(S-K,0)*np.exp(-0.5*mu**2-mu*G) #call function

    #MC_price=np.mean(payoff)

# 95% C.I

    STD=np.std(np.maximum(S-K,0)*np.exp(-0.5*mu**2-mu*G)) # standard deviation estimator

    #error=1.96*sigma/np.sqrt(Sample_size)

    #CI_up=MC_price + error
    #CI_down=MC_price -error
    
    return STD


N=100 # points for the plot
var=np.zeros(N)
i=0
V=np.linspace(-1,4,N)


for mu in V:
    
    var[i]=European_call_MC_BS_IS(0.05,100,0.2,1,100,100000,mu)
    i=i+1
    
print(var)    


plt.plot(V,var)
    