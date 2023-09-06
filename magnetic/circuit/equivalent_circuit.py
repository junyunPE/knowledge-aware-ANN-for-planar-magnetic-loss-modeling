import mph
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import ttk
import cmath
import numpy as np
import math 
import pandas as pd
import random
import tensorflow as tf
import optuna
np.random.seed(42)
tf.random.set_seed(2)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from optuna.samplers import TPESampler

import functions_circuit
import comsol_circuit


mu0=4*math.pi*1e-7
Np=4
Ns=2

def simulate(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta):
    fs = fs/1e6
    dw = dw*1e3
    thickness=thickness*1e3
    da = da*1e3
    dc = dc*1e3
    dv = dv*1e3
    dgap1=dgap1*1e3
    dgap2=dgap2*1e3
    Ta=Ta
    Z11,Z21=comsol_circuit.run_comsol_Z11_Z21('odd and open.mph', 'odd and open', fs, dw, da, dc, dv, dgap1, dgap2, Ta)
    Z22=comsol_circuit.run_comsol_Z22('odd and open secondary.mph', 'odd and open secondary', fs, dw, da, dc, dv, dgap1, dgap2, Ta)
    Z11, Z21, Z22=functions_circuit.digit_set(Z11, Z21, Z22, 4) #Four digits after the decimal point
    Z12=Z21
    return Z11,Z12,Z21,Z22

#model_dc=tf.keras.models.load_model('Rdc_model')
def calculate(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta):
    L11,L22,L21=functions_circuit.L_calculate(dgap1,dgap2,da,dc,Ns,Np)
    Rratio11,Rratio22,Rratio12=functions_circuit.R_calculate(fs,thickness)
    #Rdc=comsol_circuit.run_comsol_Rdc('odd and open(dc).mph', 'odd and open', fs, dw, da, dc, dv, dgap1, dgap2, thickness, Ta)
    input=np.array([dw*1e3, dc*1e3, da*1e3]).reshape((1, 3)) #mm
    input,Rdcmax,Rdcmin=functions_circuit.norm_Rdc('C:\\Users\\junyun_deng\\Desktop\\data\\Rdc.csv',input) #log normlization for input and output
    model_dc=tf.keras.models.load_model('Rdc_model')
    Rdc=np.exp(model_dc.predict(input).reshape(-1)*(Rdcmax-Rdcmin)+Rdcmin)
    
    R11=Rratio11*Rdc
    R22=Rratio22*Rdc
    R12=Rratio12*Rdc
    R21=R12
    
    Z11=R11+L11*2*math.pi*fs*1j
    Z22=R22+L22*2*math.pi*fs*1j
    Z21=R21+L21*2*math.pi*fs*1j
    Z11, Z21, Z22=functions_circuit.digit_set(Z11, Z21, Z22, 4)
    Z12=Z21
    return Z11,Z12,Z21,Z22
'''
model1=tf.keras.models.load_model('R11_model')
model2=tf.keras.models.load_model('R22_model')
model3=tf.keras.models.load_model('R21_model')
model4=tf.keras.models.load_model('Z11_model')
model5=tf.keras.models.load_model('Z22_model')
model6=tf.keras.models.load_model('Z21_model')
'''
def AI(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta):
    L11,L22,L21=functions_circuit.L_calculate(dgap1,dgap2,da,dc,Ns,Np)
    Rratio11,Rratio22,Rratio21=functions_circuit.R_calculate(fs,thickness)
    
    input=np.array([dw*1e3, dc*1e3, da*1e3]).reshape((1, 3)) #mm
    input,Rdcmax,Rdcmin=functions_circuit.norm_Rdc('C:\\Users\\junyun_deng\\Desktop\\data\\Rdc.csv',input) #log normlization for input and output
    model1=tf.keras.models.load_model('Rdc_model')
    Rdc=np.exp(model1.predict(input).reshape(-1)*(Rdcmax-Rdcmin)+Rdcmin)
    #Rdc=comsol_circuit.run_comsol_Rdc('odd and open(dc).mph', 'odd and open', fs, dw, da, dc, dv, dgap1, dgap2, thickness, Ta)
    Ra11=Rratio11*Rdc
    Ra22=Rratio22*Rdc
    Ra21=Rratio21*Rdc

    Za11=L11*2*math.pi*fs
    Za22=L22*2*math.pi*fs
    Za21=L21*2*math.pi*fs

    random.seed(2)
    np.random.seed(2)
    tf.random.set_seed(2)

    input=np.array([dw, da, dc, dgap1, dgap2, fs, Ra11]).reshape((1, 7))
    input,R11max,R11min=functions_circuit.norm_R('C:\\Users\\junyun_deng\\Desktop\\data\\R11.csv',input) #log normlization for input and output
    R11=np.exp(model1.predict([input[:,0:6],input[:,6]]).reshape(-1)*(R11max-R11min)+R11min)

    input=np.array([dw, da, dc, dgap1, dgap2, fs, Ra22]).reshape((1, 7))
    input,R22max,R22min=functions_circuit.norm_R('C:\\Users\\junyun_deng\\Desktop\\data\\R22.csv',input) #log normlization for input and output
    R22=np.exp(model2.predict([input[:,0:6],input[:,6]]).reshape(-1)*(R22max-R22min)+R22min)

    input=np.array([dw, da, dc, dgap1, dgap2, fs, Ra21]).reshape((1, 7))
    input,R21max,R21min=functions_circuit.norm_R('C:\\Users\\junyun_deng\\Desktop\\data\\R21.csv',input) #log normlization for input and output
    R21=np.exp(model3.predict([input[:,0:6],input[:,6]]).reshape(-1)*(R21max-R21min)+R21min)

    input=np.array([da, dc, dgap1, dgap2, fs, Za11]).reshape((1, 6))
    input,Z11max,Z11min=functions_circuit.norm_Z('C:\\Users\\junyun_deng\\Desktop\\data\\Z11.csv',input) #log normlization for input and output
    Z11=np.exp(model4.predict([input[:,0:5],input[:,5]]).reshape(-1)*(Z11max-Z11min)+Z11min)

    input=np.array([da, dc, dgap1, dgap2, fs, Za22]).reshape((1, 6))
    input,Z22max,Z22min=functions_circuit.norm_Z('C:\\Users\\junyun_deng\\Desktop\\data\\Z22.csv',input) #log normlization for input and output
    Z22=np.exp(model5.predict([input[:,0:5],input[:,5]]).reshape(-1)*(Z22max-Z22min)+Z22min)

    input=np.array([da, dc, dgap1, dgap2, fs, Za21]).reshape((1, 6))
    input,Z21max,Z21min=functions_circuit.norm_Z('C:\\Users\\junyun_deng\\Desktop\\data\\Z21.csv',input) #log normlization for input and output
    Z21=np.exp(model6.predict([input[:,0:5],input[:,5]]).reshape(-1)*(Z21max-Z21min)+Z21min)

    Z11=R11+Z11*1j
    Z21=R21+Z21*1j
    Z22=R22+Z22*1j
    Z11, Z21, Z22=functions_circuit.digit_set(Z11, Z21, Z22, 4)
    Z12=Z21
    return Z11,Z12,Z21,Z22

