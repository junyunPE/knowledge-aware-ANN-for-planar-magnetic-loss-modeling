import numpy as np
import sys
sys.path.append('D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\magnetic\\circuit')
sys.path.append('D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\magnetic\\core')
sys.path.append('D:\\file\\我的坚果云\\file\\LCT\\LLC transformer design\\software\\magnetic\\thermal')
import equivalent_circuit
import core_loss
import heat

Np=4
Ns=2
###MOO

def loss_sum_cal(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip):
    Z11,Z12,Z21,Z22=equivalent_circuit.calculate(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta)
    Is=Ip*Np/Ns
    Ppri_cal=(0.5*Z11.real*Ip*Ip+0.5*Z12.real*Ip*Is)/2
    Psec_cal=(0.5*Z21.real*Ip*Is+0.5*Z22.real*Is*Is)/2
    if  waveform=="Sinusoidal":
        Pc=core_loss.calculate_sin(fs, dw, da, dc, dv, dgap1, dgap2,  Vps)/2
    else:
        Pc=core_loss.calculate_rect(fs, dw, da, dc, dv, dgap1, dgap2, Vpr, duty_cycle)/2
    return Ppri_cal,Psec_cal,Pc

def temperature(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip):
    Ppri_cal, Psec_cal, Pc=loss_sum_cal(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
    Ppri_cal=float(Ppri_cal[0])
    Psec_cal=float(Psec_cal[0])
    Tmax_core,Tmax_pri,Tmax_sec=heat.calculate(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Ppri_cal, Psec_cal, Pc)
    Tmax_core=Tmax_core-273.15
    Tmax_pri=Tmax_pri-273.15
    Tmax_sec=Tmax_sec-273.15
    return Tmax_core,Tmax_pri,Tmax_sec

def max_Bm(fs,Vpeak,N,section_area,Vpostive,duty_cycle,waveform):
    if  waveform=="Sinusoidal":
        Bm=Vpeak/N/section_area/(2*np.pi*fs)
    else:
        Bm=Vpostive/N/section_area*duty_cycle/2/fs
    return Bm

def totol_volume(dw, da, dc, dv, dgap1):
    th_total=1.6/1e3
    bottom_solder=0.01/1e3
    Ae=np.pi*(dc/2)**2+da*da
    B=2*dw+dc+2*da
    C=Ae/2/B
    A=2*dw+dc+2*C
    I=th_total+bottom_solder+dv+dgap1
    volume=A*B*(I+2*C)
    return volume

'''
fs=500e3
Ta=25+273.15
waveform="Rectangular"
Vps=400
Vpr=400
duty_cycle=0.5
thickness=0.07/1e3
Ip=1
dw=5/1e3
da=1/1e3
dc=10/1e3
dv=1/1e3
dgap1=0.5/1e3
dgap2=0.5/1e3
print(loss_sum_cal(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip))
'''