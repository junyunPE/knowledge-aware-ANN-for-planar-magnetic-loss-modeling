import cmath
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
import functions_MOO
import matplotlib.pyplot as plt
import itertools
import numpy as np
from functools import partial
Np=4
Ns=2
'''
def analytical(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip):
    fs=500e3
    Ta=25+273.15
    waveform="Rectangular"
    Vps=400
    Vpr=400
    duty_cycle=0.5
    thickness=0.07/1e3
    class MyProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=7,
                            n_obj=2,
                            n_ieq_constr=2,
                            xl=np.array([5/1e3, 10/1e3, 1/1e3, 1/1e3, 0.25/1e3, 0.25/1e3, 1]),
                            xu=np.array([15/1e3, 35/1e3, 3/1e3, 1/1e3, 1/1e3, 1/1e3, 5]))

        def _evaluate(self, x, out, *args, **kwargs):
            dw=x[0]
            dc=x[1]
            da=x[2]
            dv=x[3]
            dgap1=x[4]
            dgap2=x[5]
            Ip=x[6]

            Ppri_cal,Psec_cal,Pc=functions_MOO.loss_sum_cal(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
            Psum=-(Ppri_cal+Psec_cal+Pc)
            volume=-(functions_MOO.totol_volume(dw, da, dc, dv, dgap1))

            Tmax_core,Tmax_pri,Tmax_sec=functions_MOO.temperature(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
            Bm=functions_MOO.max_Bm(fs,Vps,Np,np.pi*((dc/2)**2),Vpr,duty_cycle,waveform)
            g1=Tmax_core-150  
            g2=Bm-0.4

            out["F"] = [Psum, volume]
            out["G"] = [g1,g2]


    problem = MyProblem()

    algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 40)
    res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)
    X = res.X
    F = res.F

    return X,F
'''

def analytical_brutal_force(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip):

    Ppri_cal,Psec_cal,Pc=functions_MOO.loss_sum_cal(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
    Psum=Ppri_cal+Psec_cal+Pc
    Psum=float(Psum[0])
    volume=functions_MOO.totol_volume(dw, da, dc, dv, dgap1)

    Tmax_core,Tmax_pri,Tmax_sec=functions_MOO.temperature(fs, dw, thickness, da, dc, dv, dgap1, dgap2, Ta, waveform, Vps, Vpr, duty_cycle, Ip)
    Bm=functions_MOO.max_Bm(fs,Vps,Np,np.pi*((dc/2)**2),Vpr,duty_cycle,waveform)
    g1=Tmax_core-150  
    g2=Bm-0.4
    
    return Psum, volume, g1, g2

def brutal_force_multi_objective_optimization(objetives, fs, Ta, waveform, Vps, Vpr, duty_cycle, thickness, Ip):
    def multi_objective_optimization(f, bounds, step):
        n = 2 
        variables = [list(range(int((bound[1]-bound[0])/step[i]))) for i, bound in enumerate(bounds)]
        combinations = itertools.product(*variables)
        optimal_solutions = []
        solutions = []
        for combination in combinations:
            x = [bounds[i][0] + step[i] * combination[i] for i in range(len(combination))]
            Psum, volume, g1, g2 = f(dw=x[0], dc=x[1], da=x[2], dv=x[3], dgap1=x[4], dgap2=x[5])
            fx = [Psum, volume]  
            solutions.append((x, fx))
            if len(optimal_solutions) == 0:
                optimal_solutions.append((x, fx))
            else:
                dominated = False
                for i in range(len(optimal_solutions)):
                    if all(fx[j] >= optimal_solutions[i][1][j] for j in range(n)):
                        dominated = True
                        break
                    elif all(fx[j] <= optimal_solutions[i][1][j] for j in range(n)):
                        optimal_solutions.pop(i)
                        break
                if not dominated:
                    optimal_solutions.append((x, fx))
        return optimal_solutions,solutions

    f = partial(objetives, fs=fs, Ta=Ta, waveform=waveform, Vps=Vps, Vpr=Vpr, duty_cycle=duty_cycle, thickness=thickness, Ip=Ip)
    #f = partial(objetives, fs=500e3, Ta=25+273.15, waveform="Rectangular", Vps=400, Vpr=400, duty_cycle=0.5, thickness=0.07/1e3, Ip=5)

    def constraint1(x):
        return f(dw=x[0], dc=x[1], da=x[2], dv=x[3], dgap1=x[4], dgap2=x[5])[2]

    def constraint2(x):
        return f(dw=x[0], dc=x[1], da=x[2], dv=x[3], dgap1=x[4], dgap2=x[5])[3]

    bounds = [(5/1e3, 15/1e3), (20/1e3, 30/1e3), (1/1e3, 3/1e3), (1/1e3, 3/1e3), (0.2/1e3, 1/1e3), (0.2/1e3, 1/1e3)]
    step = [5e-3, 5e-3, 1e-3, 1e-3, 0.2e-3, 0.2e-3]

    optimal_solutions, solutions = multi_objective_optimization(f=f, bounds=bounds, step=step)

    solutions_filtered = []
    for sol in solutions:
        if constraint1(sol[0]) <= 0 and constraint2(sol[0]) <= 0:
            solutions_filtered.append(sol)

    optimal_solutions_filtered = []
    for sol in optimal_solutions:
        if constraint1(sol[0]) <= 0 and constraint2(sol[0]) <= 0:
            optimal_solutions_filtered.append(sol)

    print("共有 {} 个解:".format(len(solutions_filtered)))
    for sol in solutions_filtered:
        print("x={}, f(x)={}".format(sol[0], sol[1]))

    print("共有 {} 个最优解:".format(len(optimal_solutions_filtered)))
    for sol in optimal_solutions_filtered:
        print("x={}, f(x)={}".format(sol[0], sol[1]))
    
    return solutions_filtered,optimal_solutions_filtered

def analytical(fs, Ta, waveform, Vps, Vpr, duty_cycle, thickness, Ip):
    solutions_filtered,optimal_solutions_filtered=brutal_force_multi_objective_optimization(analytical_brutal_force,fs, Ta, waveform, Vps, Vpr, duty_cycle, thickness, Ip)
    return solutions_filtered,optimal_solutions_filtered




