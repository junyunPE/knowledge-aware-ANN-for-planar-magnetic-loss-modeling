import mph
import cmath
import numpy as np
import math 
import pandas as pd
import tensorflow as tf
import sys
sys.path.append('D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\magnetic\\core\\library')
import functions_core
import comsol_core
import materials

mu0=4*math.pi*1e-7
Np=4
Ns=2
k=7.16808273987195
alpha=1.3858645784187265
beta=2.7427205331725037
path='C:\\Users\\junyun_deng\\Downloads\\B.csv'
path2='C:\\Users\\junyun_deng\\Downloads\\V.csv'


def simulate_sin(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta,Vps):
    fs = fs/1e6 #MHz
    dw = dw*1e3 #mm
    thickness=thickness*1e3 #mm
    da = da*1e3 #mm
    dc = dc*1e3 #mm
    dv = dv*1e3 #mm
    dgap1=dgap1*1e3 #mm
    dgap2=dgap2*1e3 #mm

    loss=comsol_core.run_comsol_core_loss_SE('core_loss_sin.mph', 'core_loss', fs, dw, da, dc, dv, dgap1, dgap2, Ta, Vps, k, alpha, beta)
    return loss*2

def simulate_rect(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Vpr, duty_cycle):
    fs = fs/1e6 #MHz
    dw = dw*1e3 #mm
    thickness=thickness*1e3 #mm
    da = da*1e3 #mm
    dc = dc*1e3 #mm
    dv = dv*1e3 #mm
    dgap1=dgap1*1e3 #mm
    dgap2=dgap2*1e3 #mm
    comsol_core.run_comsol_core_loss_IGSE('core_loss_rect.mph', 'core_loss', fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Vpr, duty_cycle, path)
    data,volume=functions_core.readfile(path)
    k_i=functions_core.k_i(alpha,beta,k)
    time=np.linspace(start=0,stop=1/(fs*1e6),num=len(data[1,:]))
    loss= np.zeros(len(data))
    for i in range(len(data)):
        loss[i]=functions_core.loss_iGSE(fs*1e6, data[i,:], time, k_i, alpha, beta, n_interval=10_000)*volume[i]
    return np.sum(loss)*2

def calculate_sin(fs, dw, da, dc, dv, dgap1, dgap2,  Vps):
    Ve1,Ve2,Ve3,Ae=functions_core.Ve(dw, da, dc, dv, dgap1, dgap2)
    Bac=Vps/Np/Ae/(2*np.pi*fs)
    Pc=k*(fs**alpha)*(Bac**beta)*Ve1+k*(fs**alpha)*((Bac/2)**beta)*Ve2+k*(fs**alpha)*((Bac/2)**beta)*Ve3
    return Pc

def calculate_rect(fs, dw, da, dc, dv, dgap1, dgap2, Vpr, duty_cycle):
    Ve1,Ve2,Ve3,Ae=functions_core.Ve(dw, da, dc, dv, dgap1, dgap2)
    time,Voltage,B=functions_core.flux_waveform_rectangular(fs,Vpr,duty_cycle,Np,Ae)
    k_i=functions_core.ki(alpha,beta,k)
    Pc=functions_core.loss_iGSE(fs, B, time, k_i, alpha, beta, n_interval=10_000)*Ve1+functions_core.loss_iGSE(fs, B/2, time, k_i, alpha, beta, n_interval=10_000)*Ve2+functions_core.loss_iGSE(fs, B/2, time, k_i, alpha, beta, n_interval=10_000)*Ve3
    return Pc


def AI_rect(fs, dw, da, dc, dv, dgap1, dgap2, Vpr, duty_cycle,core_name):
    for component in [materials.Core_DMR53, materials.Core_3F4]:
        core = 0
        if component['name'] == core_name:
            core=component
            break
    Ve1,Ve2,Ve3,Ae=functions_core.Ve(dw, da, dc, dv, dgap1, dgap2)
    Bm=Vpr/Np/Ae*duty_cycle*1/fs/2
    Pc=core['Core loss modeling'](fs,Bm)*Ve1+core['Core loss modeling'](fs,Bm/2)*Ve2+core['Core loss modeling'](fs,Bm/2)*Ve3
    return Pc                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            



fs=np.linspace(100e3,1000e3,100)
Ta=25+273.15
waveform="Rectangular"
Vps=400
Vpr=400
duty_cycle=0.5
thickness=0.07/1e3
Ip=1
dw=5/1e3
da=1/1e3
dc=20/1e3
dv=1/1e3
dgap1=0.5/1e3
dgap2=0.5/1e3

loss = np.zeros(len(fs), dtype=np.int32)  # 创建一个包含len(fs)个元素的，数据类型为整数的零数组，用于保存所有的输出h

for i, f in enumerate(fs):
    loss[i] = AI_rect(f, dw, da, dc, dv, dgap1, dgap2, Vpr, duty_cycle,materials.Core_DMR53['name'])

print(loss)
#print(calculate_rect(fs, dw, da, dc, dv, dgap1, dgap2, Vpr, duty_cycle))
