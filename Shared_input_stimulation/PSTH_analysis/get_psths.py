import numpy as np
import pandas as pd
import struct
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
'''
#### Script designed to use 6 cores
#### Network configuration are analyzed in serie and stimuli intensitie in parallel
#### run from terminal using: 'mpirun -np 6 python get_psth.py'

Read simulations output files and get a PSTH trace per each neuron
It will create one file per network configuration and stimuli intensity
file will be saved in folder "data"
'''
################################################################
################################################################
N = 1000
################################################################
stimulis = [400, 600, 800, 1000, 1200, 1400]

################################################################
# histograms functions
def get_hist_pop(spk):  # collective PTSH
    dtbin = 0.0002  # 0.2 ms
    tbin = np.arange(lb, ub, dtbin)
    bins = len(tbin)
    return np.histogram(spk, bins=bins, range=hrange)[0] / (nrep * dtbin)
###########################################################
def get_hist(spk):  # Individual neurons PSTH
    return np.histogram(spk, bins=bins, range=hrange)[0] / (nrep * dtbin)


##################################################################
###############         path to simulations         ##############
##################################################################
path = '../PSTH_simulations/'

##################################################################
###############       Time bin for histograms       ##############
##################################################################
dtbin = 0.0005  # 0.5 ms
lb, ub = -0.025, 0.175
hrange = [lb, ub]
tbin = np.arange(lb, ub, dtbin)
bins = len(tbin)
##################################################################
###############       Barrage disconnected       #################
##################################################################
nrep = 100000
###########################################################
for kk, stim in enumerate(stimulis):
    if rank == kk:
        print('gcl', rank, kk)
        ################################################################
        hpopulation = get_hist_pop(np.ones(1000))
        hneurons = [get_hist(np.ones(1000))
                    for i in range(N)]  # N hist fill with zeros
        for k, proc in enumerate(range(10)):
            print('gcl', rank, kk, k)
            folder = '%s/psth_gcl/psth_gcl_%d_%d' % (path, kk, k)
            ########################
            f = open('%s/Spikes_times.dat' %
                     folder, mode='rb').read()  # read spikes file
            format = '%df' % (len(f) // 4)
            spk = np.array(struct.unpack(format, f))
            hpopulation += get_hist_pop(spk)
            #################
            spkn = np.array(pd.read_csv('%s/Spiking_neurons.dat' %
                                        folder, sep='\s+', header=None))
            spkn = spkn.reshape(len(spkn))

            for j in range(N):
                hneurons[j] += get_hist(spk[np.where(spkn == j + 1)])
        np.save('data/hist_gcl_%d.npy' % kk, hneurons)
        # np.save('data/hist_gcl_pop_%d.npy' % kk, hpopulation / N)

##################################################################
###############       Random Network       #######################
##################################################################
nrep = 100000
###########################################################
for kk, stim in enumerate(stimulis):
    if rank == kk:
        print('ran', rank, kk)
    ################################################################
        hpopulation = get_hist_pop(np.ones(1000))
        hneurons = [get_hist(np.ones(1000))
                    for i in range(N)]  # N hist fill with zeros
        for k, proc in enumerate(range(10)):
            print('ran', rank, kk, k)
            folder = '%s/psth_ran/psth_ran_%d_%d' % (path, kk, k)
            ########################
            f = open('%s/Spikes_times.dat' %
                     folder, mode='rb').read()  # read spikes file
            format = '%df' % (len(f) // 4)
            spk = np.array(struct.unpack(format, f))
            hpopulation += get_hist_pop(spk)
            #################
            spkn = np.array(pd.read_csv('%s/Spiking_neurons.dat' %
                                        folder, sep='\s+', header=None))
            spkn = spkn.reshape(len(spkn))

            for j in range(N):
                hneurons[j] += get_hist(spk[np.where(spkn == j + 1)])
        np.save('data/hist_ran_%d.npy' % kk, hneurons)
        # np.save('data/hist_ran_pop_%d.npy' % kk, hpopulation / N)

##################################################################
##############       Disconnected no barrage       ##############
##################################################################
nrep = 100000
###########################################################
for kk, stim in enumerate(stimulis):
    if rank == kk:
        print('con', rank, kk)
        ################################################################
        hpopulation = get_hist_pop(np.ones(1000))
        hneurons = [get_hist(np.ones(1000))
                    for i in range(N)]  # N hist fill with zeros
        for k, proc in enumerate(range(10)):
            print('con', rank, kk, k)
            folder = '%s/psth_con/psth_con_%d_%d' % (path, kk, k)
            ########################
            f = open('%s/Spikes_times.dat' %
                     folder, mode='rb').read()  # read spikes file
            format = '%df' % (len(f) // 4)
            spk = np.array(struct.unpack(format, f))
            hpopulation += get_hist_pop(spk)
            #################
            spkn = np.array(pd.read_csv('%s/Spiking_neurons.dat' %
                                        folder, sep='\s+', header=None))
            spkn = spkn.reshape(len(spkn))

            for j in range(N):
                hneurons[j] += get_hist(spk[np.where(spkn == j + 1)])
        np.save('data/hist_con_%d.npy' % kk, hneurons)
        # np.save('data/hist_con_pop_%d.npy' % kk, hpopulation / N)


##################################################################
###############       Small-world Network       ##################
##################################################################
nrep = 100000
###########################################################
for kk, stim in enumerate(stimulis):
    if rank == kk:
        print('net', rank, kk)
        ################################################################
        hpopulation = get_hist_pop(np.ones(1000))
        hneurons = [get_hist(np.ones(1000))
                    for i in range(N)]  # N hist fill with zeros
        for k, proc in enumerate(range(10)):
            print('net', rank, kk, k)
            folder = '%s/psth_net/psth_net_%d_%d' % (path, kk, k)
            ########################
            f = open('%s/Spikes_times.dat' %
                     folder, mode='rb').read()  # read spikes file
            format = '%df' % (len(f) // 4)
            spk = np.array(struct.unpack(format, f))
            hpopulation += get_hist_pop(spk)
            #################
            spkn = np.array(pd.read_csv('%s/Spiking_neurons.dat' %
                                        folder, sep='\s+', header=None))
            spkn = spkn.reshape(len(spkn))

            for j in range(N):
                hneurons[j] += get_hist(spk[np.where(spkn == j + 1)])
        np.save('data/hist_net_%d.npy' % kk, hneurons)
        # np.save('data/hist_net_pop_%d.npy' % kk, hpopulation / N)
#     ################################################################
