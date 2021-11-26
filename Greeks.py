#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:55:19 2021

@author: kebaiera
"""


import numpy as np   # math operations
import numpy.random as npr # random 
import matplotlib.pyplot as plt # plot

from scipy.stats import norm




#######################
###  Bump Method  #####
#######################


def Delta_call_MC_BS_centred(r,S0,sigma,T,K,Tol):
    
   
    
    #parametrezation for a centred scheme
    epsilon=np.sqrt(Tol)
    Sample_size=int(1/(Tol**2)) 
    
    
    G=npr.normal(0,1,size=(1,Sample_size))
    
    S_back=(S0-epsilon)*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*G)
    payoff_back=np.exp(-r*T)*np.maximum(S_back-K,0) #call function
    
    S_for=(S0+epsilon)*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*G)
    payoff_for=np.exp(-r*T)*np.maximum(S_for-K,0)
    
    Delta_cent=(payoff_for -payoff_back)/(2*epsilon)
    


    #MC_price=np.mean(payoff)
    
    delta_price=np.mean(Delta_cent)

# 95% C.I

    STD=np.std(Delta_cent) # standard deviation estimator

    error_delta=1.96*STD/np.sqrt(Sample_size)
    
    d=1/(sigma*np.sqrt(T))*(np.log(K/S0)-(r-0.5*sigma**2)*T)
    d1=-d+sigma*np.sqrt(T)
    
    
    True_delta=norm.cdf(d1)


    return  delta_price,True_delta





N=1000
MSE=0

for k in range(N):
    [x,y]=Delta_call_MC_BS_centred(0.05,100,0.20,1,100,0.01)
    MSE=MSE+(x-y)**2
    print(k)
    
MSE=MSE/N
RMSE=np.sqrt(MSE)    


print(RMSE)

###########################
###  Pathwise Method  #####
###########################


def Delta_European_call_MC_BS_Path(r,S0,sigma,T,K,Sample_size):
    
    G=npr.normal(0,1,size=(1,Sample_size))
    S=S0*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*G)
    
    Delta_payoff=np.exp(-r*T)*S/S0*(S>=K)
    
    Delta=np.mean(Delta_payoff)
    
    STD=np.std(Delta_payoff) # standard deviation estimator



    error_delta=1.96*STD/np.sqrt(Sample_size)
    
    CI_up=Delta+error_delta
    CI_down=Delta-error_delta
    
    d=1/(sigma*np.sqrt(T))*(np.log(K/S0)-(r-0.5*sigma**2)*T)
    d1=-d+sigma*np.sqrt(T)
    
    
    True_delta=norm.cdf(d1)
    
    return Delta, error_delta, CI_up,CI_down,True_delta


[Delta, error_delta, CI_up,CI_down,True_delta]=Delta_European_call_MC_BS_Path(0.05,100,0.2,1,100,2000000)

print(Delta,error_delta, CI_up,CI_down,True_delta)


def Vega_European_call_MC_BS_Path(r,S0,sigma,T,K,Sample_size):
    
    G=npr.normal(0,1,size=(1,Sample_size))
    S=S0*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*G)
    
    Vega_payoff=np.exp(-r*T)*(np.sqrt(T)*G-sigma*T)*S*(S>=K)
    
    Vega=np.mean(Vega_payoff)
    
    STD=np.std(Vega_payoff) # standard deviation estimator



    error_Vega=1.96*STD/np.sqrt(Sample_size)
    
    CI_up_V=Vega+error_Vega
    CI_down_V=Vega-error_Vega
    
    
    return Vega, error_Vega, CI_up_V, CI_down_V


[Vega, error_Vega, CI_up_V, CI_down_V]=Vega_European_call_MC_BS_Path(0.05,100,0.2,1,100,2000000)

print('***********')
print(Vega, error_Vega, CI_up_V, CI_down_V)

###Asian Option 

def Delta_Asian_call_MC_BS(r,S0,sigma,T,K,N,n):
    
    delta=float(T/n)

    G=npr.normal(0,1,size=(N,n))

    #Log returns
    LR=(r-0.5*sigma**2)*delta+np.sqrt(delta)*sigma*G
    # concatenate with log(S0)
    LR=np.concatenate((np.log(S0)*np.ones((N,1)),LR),axis=1)
    # cumsum horizontally (axis=1)
    LR=np.cumsum(LR,axis=1)
    # take the expo Spath matrix
    Spaths=np.exp(LR)
    #print(Spaths)
    
    ## Riemann approximation for the continuous Asian option
    #remove final time component
    #Spaths=Spaths[:,0:len(Spaths[0,:])-1]
    #print(Spaths)
    
    ## Discrete Asian option over n future dates (St1+St2+....+Stn/n)
    #remove first time component
    Spaths=Spaths[:,1:len(Spaths[0,:])]
    #print(Spaths)
    
    
    #take the average over each row
    Sbar=np.mean(Spaths,axis=1)
    #print(Sbar)
    
    Delta_payoff_A=np.exp(-r*T)*(Sbar>=K)*Sbar/S0 #call function

    Delta_Asian_MC_price=np.mean(Delta_payoff_A)

    # 95% C.I

    STD=np.std(Delta_payoff_A) # standard deviation estimator

    error_Delta_A=1.96*STD/np.sqrt(N)

    CI_up_Delta_A=Delta_Asian_MC_price + error_Delta_A
    CI_down_Delta_A=Delta_Asian_MC_price -error_Delta_A
    
    return Delta_Asian_MC_price,CI_up_Delta_A,CI_down_Delta_A,error_Delta_A




[Delta_Asian_MC_price,CI_up_Delta_A,CI_down_Delta_A,error_Delta_A]=Delta_Asian_call_MC_BS(0.05,100,0.2,1,100,100000,10)    
print('*******Delta Asian CAll Pathwise method*****')
print(Delta_Asian_MC_price,CI_up_Delta_A,CI_down_Delta_A,error_Delta_A)

###################################
### Log Likelihood ratio method####
###################################



def Delta_LLR_European_call_MC_BS(r,S0,sigma,T,K,Sample_size):
    
    G=npr.normal(0,1,size=(1,Sample_size))
    S=S0*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*G)
    
    LLR_payoff=np.exp(-r*T)*np.maximum(S-K,0)*G/(S0*sigma*np.sqrt(T)) #call function

    LLR_Delta=np.mean(LLR_payoff)

# 95% C.I

    STD=np.std(LLR_payoff) # standard deviation estimator

    error=1.96*STD/np.sqrt(Sample_size)

    CI_up=LLR_Delta + error
    CI_down=LLR_Delta -error
    
    return LLR_Delta,CI_up,CI_down,error




[LLR_Delta,CI_up,CI_down,error]=Delta_LLR_European_call_MC_BS(0.05,100,0.2,1,100,10000000)    
print('*******DELTA_LLRM*****')
print(LLR_Delta,CI_up,CI_down,error)



def GAMMA_LLR_PW_European_call_MC_BS(r,S0,sigma,T,K,Sample_size):
    
    G=npr.normal(0,1,size=(1,Sample_size))
    S=S0*np.exp((r-sigma**2/2)*T+sigma*np.sqrt(T)*G)
    
    LLR_PW_payoff=np.exp(-r*T)*G/(S0**2*sigma*np.sqrt(T))*((S>=K)*S-np.maximum(S-K,0)) #call function

    LLR_PW_Gamma=np.mean(LLR_PW_payoff)

# 95% C.I

    STD=np.std(LLR_PW_payoff) # standard deviation estimator

    error=1.96*STD/np.sqrt(Sample_size)

    CI_up=LLR_PW_Gamma + error
    CI_down=LLR_PW_Gamma -error
    
    d=1/(sigma*np.sqrt(T))*(np.log(K/S0)-(r-0.5*sigma**2)*T)
    d1=-d+sigma*np.sqrt(T)
    
    
    True_Gamma=norm.pdf(d1)/(S0*sigma*np.sqrt(T))
    
    
    return LLR_PW_Gamma,CI_up,CI_down,error,True_Gamma




[LLR_PW_Gamma,CI_up,CI_down,error,True_Gamma]=GAMMA_LLR_PW_European_call_MC_BS(0.05,100,0.2,1,100,10000000)    
print('*******Gamma_LLR_PW*****')
print(LLR_PW_Gamma,CI_up,CI_down,error,True_Gamma)



######################
# Asian Options LLR
######################


def Delta_Asian_call_MC_BS_LLR(r,S0,sigma,T,K,N,n):
    
    delta=float(T/n)

    G=npr.normal(0,1,size=(N,n))

    #Log returns
    LR=(r-0.5*sigma**2)*delta+np.sqrt(delta)*sigma*G
    # concatenate with log(S0)
    LR=np.concatenate((np.log(S0)*np.ones((N,1)),LR),axis=1)
    # cumsum horizontally (axis=1)
    LR=np.cumsum(LR,axis=1)
    # take the expo Spath matrix
    Spaths=np.exp(LR)
    #print(Spaths)
    
    ## Discrete Asian option over n future dates (St1+St2+....+Stn/n)
    #remove first time component
    Spaths=Spaths[:,1:len(Spaths[0,:])]
    #print(Spaths)
    
    #take the average over each row
    Sbar=np.mean(Spaths,axis=1)
   
    Derivative_Weight=G[:,0]/(S0*sigma*np.sqrt(delta))
    
    # Delta payoff by LLR method
    Delta_payoff_LLR=np.exp(-r*T)*np.maximum(Sbar-K,0)*Derivative_Weight 
    
    Delta_Asian_MC_price_LLR=np.mean(Delta_payoff_LLR)

    # 95% C.I

    STD=np.std(Delta_payoff_LLR) # standard deviation estimator

    error=1.96*STD/np.sqrt(N)

    CI_up=Delta_Asian_MC_price_LLR + error
    CI_down=Delta_Asian_MC_price_LLR -error
    
    return Delta_Asian_MC_price_LLR,CI_up,CI_down,error




[Delta_Asian_MC_price_LLR,CI_up,CI_down,error]=Delta_Asian_call_MC_BS_LLR(0.05,100,0.2,1,100,100000,10)    
print('*******Delta Asian CAll LLR *****')
print(Delta_Asian_MC_price_LLR,CI_up,CI_down,error)


