import cmath
import numpy as np
import math 
import pandas as pd

mu0=4*math.pi*1e-7

def norm_Rdc(path,input):
    temp=pd.read_csv(path,header=0)
    data=np.array(temp)
    x=data[:,0:3]
    y=data[:,3]
    ymax=np.max(np.log(y))
    ymin=np.min(np.log(y))
    input=(np.log(input) - np.min(np.log(x), axis=0)) / (np.max(np.log(x), axis=0) - np.min(np.log(x), axis=0))
    return input,ymax,ymin


def norm_R(path,input):
    temp=pd.read_csv(path,header=0)
    data=np.array(temp)
    x=data[:,1:7]
    xn=data[:,7]
    y=data[:,8]
    xn=xn.reshape((-1, 1))
    x=np.concatenate((x, xn), axis=1)

    ymax=np.max(np.log(y))
    ymin=np.min(np.log(y))

    input=(np.log(input) - np.min(np.log(x), axis=0)) / (np.max(np.log(x), axis=0) - np.min(np.log(x), axis=0))

    return input,ymax,ymin


def norm_Z(path,input):
    temp=pd.read_csv(path,header=0)
    data=np.array(temp)
    x=data[:,2:7]
    xn=data[:,7]
    y=data[:,8]
    xn=xn.reshape((-1, 1))
    x=np.concatenate((x, xn), axis=1)

    input=(np.log(input) - np.min(np.log(x), axis=0)) / (np.max(np.log(x), axis=0) - np.min(np.log(x), axis=0))
    ymax=np.max(np.log(y))
    ymin=np.min(np.log(y))

    return input,ymax,ymin

def R_calculate(fs,thickness):
    #skin depth calculation
    p=1/5.998e7    #electrical resistance of copper
    mu=1    #relative permeability of copper
    delta=math.sqrt(p/(math.pi*fs*mu*mu0))

    temp=thickness/delta
    F=(math.sinh(temp)+math.sin(temp))/(math.cosh(temp)-math.cos(temp))
    G=(math.sinh(temp)-math.sin(temp))/(math.cosh(temp)+math.cos(temp))
    Rratio11=temp/2*(4*F+52*G)
    Rratio22=temp/2*(2*F+2*G)
    Rratio12=temp/4*(16*G)
    return Rratio11,Rratio22,Rratio12

def L_calculate(dgap1,dgap2,da,dc,Ns,Np):
    RL=dgap2/(mu0*da*da ) 
    Rm=dgap1/(mu0*dc/2*dc/2*np.pi) 
    Lk=Np*Np/RL
    Lm=Np*Np/Rm
    k=Lm/(Lk+Lm)
    L11=Lm/k
    L22=L11/(Np/Ns*Np/Ns)
    L12=k*np.sqrt(L11*L22)
    return L11,L22,L12  

def digit_set(Z11, Z21, Z22, n):
    Z11 = np.round(Z11.real, n) + np.round(Z11.imag, n) * 1j
    Z21 = np.round(Z21.real, n) + np.round(Z21.imag, n) * 1j
    Z22 = np.round(Z22.real, n) + np.round(Z22.imag, n) * 1j
    return Z11,Z21,Z22

