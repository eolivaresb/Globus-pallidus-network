import os
import numpy as np
import pandas as pd
################################################################
tbin = 10/1000 # 10 miliseconds
N = 1000
ttot = 1800

'''
Script to simulate neural network,
Spikes are loaded and analized (get rates and CV for each neuron)
Pearson correlation matrix is calculated afterward
'''

def corr_matrix_output(data):
    M = np.zeros((N, int(ttot/tbin)+1))
    for spk in data:
        M[int(spk[1])-1, int(spk[0]/tbin)] +=1
    ### Correlation matrix
    M = M - M.mean(axis=1, keepdims=True)
    corrM = np.dot(M, M.T)
    for i in range(N):
        if corrM[i, i] == 0:
            corrM[i, i] = 1
    selfcorr = np.copy(np.diag(corrM))
    corrM = corrM / np.sqrt(selfcorr)[:,None]
    corrM = (corrM.T / np.sqrt(selfcorr)[:,None]).T
    return corrM

################################################################
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
simulate = 1
################################################################
if rank == 0:
    os.chdir('small')
    if simulate:
        os.system('make clean;make')
        os.system('./main')
    data = np.array(pd.read_csv('./Spikes.dat', sep='\s+', header=None))
    spikes = [data[np.where(data[:,1] == i), 0][0] for i in range(1, N+1)]
    fr = np.array([len(s)/ttot for s in spikes])
    cvisi = np.zeros(N)
    for i, s in enumerate(spikes):
        cvisi[i] = (np.std(np.diff(s))/np.mean(np.diff(s)) if (len(s)>2) else 0)
    np.savetxt('rates_and_cv.dat', np.array([fr, cvisi]), fmt = '%.3f')

    SCcorr = corr_matrix_output(data)
    np.save('corr.npy', SCcorr)
    os.chdir('../')
################################################################
if rank == 1:
    os.chdir('random')
    if simulate:
        os.system('make clean;make')
        os.system('./main')
    data = np.array(pd.read_csv('./Spikes.dat', sep='\s+', header=None))
    spikes = [data[np.where(data[:,1] == i), 0][0] for i in range(1, N+1)]
    fr = np.array([len(s)/ttot for s in spikes])
    cvisi = np.zeros(N)
    for i, s in enumerate(spikes):
        cvisi[i] = (np.std(np.diff(s))/np.mean(np.diff(s)) if (len(s)>2) else 0)
    np.savetxt('rates_and_cv.dat', np.array([fr, cvisi]), fmt = '%.3f')

    SCcorr = corr_matrix_output(data)
    np.save('corr.npy', SCcorr)
    os.chdir('../')
################################################################
