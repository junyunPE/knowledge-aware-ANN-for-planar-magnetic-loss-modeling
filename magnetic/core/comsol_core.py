import mph
import numpy as np
import cmath
import matplotlib.pyplot as plt


def run_comsol_core_loss_IGSE(simulation_name, study_name, fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, Vpr, duty_cycle, path):
    client = mph.start(cores=1)
    model = client.load(simulation_name)
    model.parameter('fs', f'{fs} [MHz]')
    model.parameter('dw', f'{dw} [mm]')
    model.parameter('thickness', f'{thickness} [mm]')
    model.parameter('F', f'{da} [mm]')
    model.parameter('D', f'{dc} [mm]')
    model.parameter('dv', f'{dv} [mm]')
    model.parameter('dgap1',f'{dgap1} [mm]')
    model.parameter('dgap2', f'{dgap2} [mm]')
    model.parameter('Ta', f'{Ta+273.15} [K]')
    model.parameter('Vpr', f'{Vpr}')
    model.parameter('duty_cycle', f'{duty_cycle}')
    model.solve(study_name)
    datasets=model/'datasets'
    (datasets/'core_loss//解 1').select(model/'selections'/'core')
    exportnode = model/'exports'
    exportnode.create("Data", name="data")
    datanode = exportnode/"data"
    datanode.property('expr', ['mf.normB','meshvol'])
    datanode.property('unit', ['T','m^3'])
    datanode.property('descr', ['flux density','Element volume'])
    model.export(datanode, path)
    client.remove(model)

def run_comsol_core_loss_SE(simulation_name, study_name, fs, dw, da, dc, dv, dgap1, dgap2, Ta, Vps, k, alpha, beta):
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
    model.parameter('Vps', f'{Vps} [V]')
    model.parameter('k', f'{k}')
    model.parameter('alpha', f'{alpha}')
    model.parameter('beta', f'{beta}')
    model.solve(study_name)
    loss=model.evaluate('core_Stmz')
    client.remove(model)
    return loss


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




