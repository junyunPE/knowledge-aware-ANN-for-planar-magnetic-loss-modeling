import mph
import numpy as np
import cmath
import matplotlib.pyplot as plt

def run_comsol_Z11_Z21(simulation_name, study_name, fs, dw, da, dc, dv, dgap1, dgap2, Ta):
    client = mph.start(cores=1)
    model = client.load(simulation_name)
    model.parameter('fs', f'{fs} [MHz]')
    model.parameter('dw', f'{dw} [mm]')
    model.parameter('F', f'{da} [mm]')
    model.parameter('D', f'{dc} [mm]')
    model.parameter('dv', f'{dv} [mm]')
    model.parameter('dgap1',f'{dgap1} [mm]')
    model.parameter('dgap2', f'{dgap2} [mm]')
    model.parameter('Ta', f'{Ta+273.15} [K]')
    model.solve(study_name)
    B=model.evaluate('Vpri/mf.ICoil_p1')
    Z11=model.evaluate('Vpri/mf.ICoil_p1')
    Z21=model.evaluate('-(mf.VCoil_s1+mf.VCoil_s2)/mf.ICoil_p1')
    Z11 = np.round(Z11.real, 4) + np.round(Z11.imag, 4) * 1j
    Z21 = np.round(Z21.real, 4) + np.round(Z21.imag, 4) * 1j
    client.remove(model)
    return Z11,Z21

def run_comsol_Z22(simulation_name, study_name, fs, dw, da, dc, dv, dgap1, dgap2, Ta):
    client = mph.start(cores=1)
    model = client.load(simulation_name)
    model.parameter('fs', f'{fs} [MHz]')
    model.parameter('dw', f'{dw} [mm]')
    model.parameter('F', f'{da} [mm]')
    model.parameter('D', f'{dc} [mm]')
    model.parameter('dv', f'{dv} [mm]')
    model.parameter('dgap1',f'{dgap1} [mm]')
    model.parameter('dgap2', f'{dgap2} [mm]')
    model.parameter('Ta', f'{Ta+273.15} [K]')
    model.solve(study_name)
    Z22=model.evaluate('Vpri/mf3.ICoil_5')
    Z22 = np.round(Z22.real, 4) + np.round(Z22.imag, 4) * 1j
    client.remove(model)
    return Z22

def run_comsol_Rdc(simulation_name, study_name, fs, dw, da, dc, dv, dgap1, dgap2, thickness, Ta):
    client = mph.start(cores=1)
    model = client.load(simulation_name)
    model.parameter('fs', f'{fs/1e6} [MHz]')
    model.parameter('dw', f'{dw*1e3} [mm]')
    model.parameter('thickness', f'{thickness*1e3} [mm]')
    model.parameter('F', f'{da*1e3} [mm]')
    model.parameter('D', f'{dc*1e3} [mm]')
    model.parameter('dv', f'{dv*1e3} [mm]')
    model.parameter('dgap1',f'{dgap1*1e3} [mm]')
    model.parameter('dgap2', f'{dgap2*1e3} [mm]')
    model.parameter('Ta', f'{Ta+273.15} [K]')
    model.solve(study_name)
    Rdc=model.evaluate('mf.RCoil_p1')
    client.remove(model)
    return Rdc

'''
fs=1
dw=5
da=3
dc=10
dv=0.2
dgap1=0.2
dgap2=0.2
Ta=25
Vpr=400
duty_cycle=0.5
thickness=0.07
Ppri=0.02
Psec=0.2
Pc=3
#path='C:\\Users\\junyun_deng\\Downloads\\B2.csv'
#run_comsol_core_loss_IGSE('core_loss_rect.mph','core_loss', fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Vpr, duty_cycle, path)
#run_comsol_heat('heat.mph','heat', fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Ppri, Psec, Pc)
'''




