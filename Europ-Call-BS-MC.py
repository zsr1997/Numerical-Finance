#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:15:29 2021

@author: kebaiera
"""

import numpy as np
import numpy.random as npr 
import matplotlib.pyplot as plt
from scipy.stats import norm


def European_call_MC_BS(r,S0,sigma,T,K,Sample_size):
    
    G=npr.normal(0,1,size=(1,Sample_size))
    S=S0*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*G)  # WT=np.sqrt(T)*G
    
    payoff=np.exp(-r*T)*np.maximum(S-K,0) #call function

    MC_price=np.mean(payoff)

#     # 95% C.I

    STD=np.std(payoff) # standard deviation estimator

    error=1.96*STD/np.sqrt(Sample_size)

    CI_up=MC_price + error
    CI_down=MC_price -error
    
    
    # True price by Black-Scholes formula 


    d1= 1./(sigma*np.sqrt(T))*(np.log(S0/K)+(r+sigma**2/2)*T)
    d2= 1./(sigma*np.sqrt(T))*(np.log(S0/K)+(r-sigma**2/2)*T)
    True_price= S0*norm.cdf(d1) -K*np.exp(-r*T)*norm.cdf(d2)
    
    return MC_price, True_price, CI_up, CI_down, error

[MC_price, True_price,CI_up, CI_down,error]=European_call_MC_BS(0.04,100,0.2,1,100,100000000)

print("Monte Carlo Call Price", MC_price)
print("True Call Price", True_price)
print("Confidence Interval up", CI_up)
print("Confidence Interval down", CI_down)
print("error", error)