import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate
import os
import pandas as pd

simulate_cond = 1
simulate_model_prcs = 1
#################################################################
def prc_eval(x, c1, c2, c3, e1, e2, e3, p_thre):
    comp1 = (x**e1*(p_thre**e1 - x**e1))/(e1*p_thre**(1+2*e1)/(1 + 3*e1+2*e1**2))
    comp2 = (x**e2*(p_thre**e2 - x**e2))/(e2*p_thre**(1+2*e2)/(1 + 3*e2+2*e2**2))
    comp3 = (x**e3*(p_thre**e3 - x**e3))/(e3*p_thre**(1+2*e3)/(1 + 3*e3+2*e3**2))
    y = c1*comp1 + c2*comp2 + c3*comp3
    return np.max([0, y])
##########################################################
### create ttot seconds of synaptic conductnace generate by a poisson process
dt = 0.0001
ttot = 1800
gamp = 2500
gstd = 1750
tau = 2.4/1000
rate = 180
Erev = -0.073
time = np.arange(0, ttot, dt)

#######################################
tnext = 0.0
'''
Generate a conductance trace from a poisson process rate = 180 Hz (aprox 10 neurons at 18 Hz each)
'''
if simulate_cond:
    g = np.zeros(len(time))
    s = 0
    for k, t in enumerate(time[1:]):
        while (tnext < dt):
            tnext += -np.log(np.random.uniform())/rate
            gevent = np.random.normal(gamp, gstd)
            if (gevent > 0): g[k-1] += gevent
            s+=1
        tnext -= dt
        g[k] = g[k-1]*(1-dt/tau)
    print ('mean rate = %.3f'%(s/ttot))
    np.save('poisson_cond.npy', g)
g = np.load('poisson_cond.npy', allow_pickle = 'true')

path_to_files = './../simulation_files'
volt = np.loadtxt('%s/volt.dat'%path_to_files)
prcfit = np.loadtxt('%s/PRCs/params_prc.txt'%path_to_files)
#################################################################
xphase = volt[:,0]
v = volt[:,1]
volt_eval = interpolate.interp1d(xphase,v)
#########################   Simulation   #############################
w = 30.46
n1, n2, n3 = 1, 10, 16 ## three neurons from experimental sample
neurons = [n1, n2, n3]
z1 = lambda x: prc_eval(x, *prcfit.T[n1])
z2 = lambda x: prc_eval(x, *prcfit.T[n2])
z3 = lambda x: prc_eval(x, *prcfit.T[n3])
pdens = [np.zeros(1000) for i in range(3)]
###################
if simulate_model_prcs:
    spikes = [[] for i in range(3)]
    phases = [np.zeros(len(time)) for i in range(3)]
    for p in phases: p[0] = np.random.rand()
    for k, t in enumerate(time[1:]):
        phases[0][k] = phases[0][k-1] + dt*(w + z1(phases[0][k-1])*g[k]*(Erev-volt_eval(phases[0][k-1])))
        phases[1][k] = phases[1][k-1] + dt*(w + z2(phases[1][k-1])*g[k]*(Erev-volt_eval(phases[1][k-1])))
        phases[2][k] = phases[2][k-1] + dt*(w + z3(phases[2][k-1])*g[k]*(Erev-volt_eval(phases[2][k-1])))
        for i in range(3):
            if (phases[i][k]>=1):
                spikes[i].append(t + dt*(1-phases[i][k]))
                phases[i][k]-=1
            if (phases[i][k]< 0): phases[i][k] = 0
            pdens[i][int(1000*phases[i][k])]+=1
    for i in range(3): phases[i] = phases[i][:20000]
    np.save('data_prcs.npy', [phases, spikes, pdens])
#########################   Simulation   #############################
os.system('python figure4.py')
