import numpy as np
import pandas as pd
import struct
import os
################################################################
'''
Read simulations output and merge trails in one file
'''

##################################################################
###############         path to simulations         ##############
##################################################################
path = '../PSTH_simulations/'

################################################################
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
################################################################
import matplotlib
matplotlib.rcParams.update(
    {'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix', })
################################################################
################################################################
N = 1000
################################################################
stimulis = [400, 600, 800, 1000, 1200, 1400]
################################################################
################################################################
dtbin = 0.0001  # 4 ms
nrep = 100000.
lb, ub = -0.025, 0.175
hrange = [lb, ub]
tbin = np.arange(lb, ub, dtbin)
bins = len(tbin)
################################################################
for kk, stim in enumerate(stimulis):
    print(kk)
    ################################################################
    Conductance = np.zeros((N, bins))
    for k, proc in enumerate(range(10)):
        print(kk, k)
        folder = 'simulations/psth_net/psth_net_%d_%d' % (kk, k)
        ########################
        ########################
        Conductance += np.array(pd.read_csv('%s/cond.dat' %
                                    folder, sep='\s+', header=None))
    np.save('data/cond_net_%d.npy' % kk, Conductance/nrep)
    # np.savetxt('data/cond_net_%d.dat' % kk, Conductance/nrep)
    ################################################################
