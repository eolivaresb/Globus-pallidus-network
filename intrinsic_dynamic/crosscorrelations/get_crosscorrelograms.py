import numpy as np
import pandas as pd
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
################################################################
################################################################
N = 1000
ttot = 1800
twindow, tbin = 0.5, 0.001
def crosscorrelation(ttot, twindow, tbin, spk1, spk2):
    bined2 = np.zeros(int(ttot/tbin) + 1)
    for s in spk2:
        bined2[int(s/tbin)] +=1
    cc = np.zeros(int(2*twindow/tbin) +1)
    for s in spk1:
        if ((s>twindow) and (s< ttot - twindow)):
            cc += bined2[int((s-twindow)/tbin): int((s-twindow)/tbin) +int(2*twindow/tbin) +1]
    cc = np.sqrt(cc/(ttot*tbin)) ## geometric mean normalization
    return cc
################################################################
if rank == 0:
    network = 'random'
    ###############################################################
    spk = np.array(pd.read_csv('../%s/Spikes.dat'%network, sep='\s+', header=None))
    spikes = [np.empty(0) for i in range(N)]
    for s in spk:
        spikes[int(s[1]-1)] = np.append(spikes[int(s[1]-1)], s[0])
    np.save('../%s/spikes_random.npy'%network, spikes)
    ################################
    spikes = np.load('../%s/spikes_random.npy'%network, allow_pickle = 'true')
    rates = np.array([len(spikes[i])/ttot for i in range(N)])
    cc = []
    for i in range(N):
        print(i)
        for j in range(i, N):
            if ((rates[i]>2.)*(rates[j]>2.)):
                cc.append(crosscorrelation(ttot, twindow, tbin, spikes[i], spikes[j]))
            else:
                cc.append([rates[i], rates[j]])
    np.save('cc_%s.npy'%network, cc)
################################################################
################################################################
if rank == 1:
    network = 'small'
    ###############################################################
    spk = np.array(pd.read_csv('../%s/Spikes.dat'%network, sep='\s+', header=None))
    spikes = [np.empty(0) for i in range(N)]
    for s in spk:
        spikes[int(s[1]-1)] = np.append(spikes[int(s[1]-1)], s[0])
    np.save('../%s/spikes_small.npy'%network, spikes)
    ################################
    spikes = np.load('../%s/spikes_small.npy'%network, allow_pickle = 'true')
    rates = np.array([len(spikes[i])/ttot for i in range(N)])
    cc = []
    for i in range(N):
        print(i)
        for j in range(i, N):
            if ((rates[i]>2.)*(rates[j]>2.)):
                cc.append(crosscorrelation(ttot, twindow, tbin, spikes[i], spikes[j]))
            else:
                cc.append([rates[i], rates[j]])
    np.save('cc_%s.npy'%network, cc)
################################################################
