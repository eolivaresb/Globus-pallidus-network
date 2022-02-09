import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
################################################################
tbin = 10/1000 # 10 miliseconds
N = 1000
'''
Script to get the connectivity relationship between pairs of neurons
It uses the connectivity matrix and goes over each pair checking the connectivity class
1: Mutually inhibited
2: directly inhibited (Not 1)
3: Disynaptic (not 1 or 2)
4: shared input from another neuron (nor 1, 2, or 3)
5: Other, none of the above
'''
################################################################
z = np.loadtxt('../../simulation_files/network_matrix_small.dat').astype(int)
anatomy = np.zeros((N, N)) #default 0 if i=j
for i in range(N):
    for j in range(i+1, N):
        if ((z[i,j]==1)*(z[j,i]==1)):  # i to j and j to i
            anatomy[i, j] = 1
            anatomy[j, i] = 1
        elif ((z[i,j]==1)+(z[j,i]==1)): # at least one connection between i and j
            anatomy[i, j] = 2
            anatomy[j, i] = 2
        elif((np.sum(z[i]*z[:,j])+np.sum(z[:,i]*z[j]))): # one neuron connect i to j or j to i
            anatomy[i, j] = 3
            anatomy[j, i] = 3
        elif(np.sum(z[:,i]*z[:,j])): #at least one neuron project to both i, j
            anatomy[i, j] = 4
            anatomy[j, i] = 4
        else:                               # none of the above categories
            anatomy[i, j] = 5
            anatomy[j, i] = 5
np.savetxt('data/anatomy_categories_small.txt', anatomy, fmt='%d')
################################################################
################################################################
z = np.loadtxt('../../simulation_files/network_matrix_random.dat').astype(int)
anatomy = np.zeros((N, N)) #default 0 if i=j
for i in range(N):
    for j in range(i+1, N):
        if ((z[i,j]==1)*(z[j,i]==1)):  # i to j and j to i
            anatomy[i, j] = 1
            anatomy[j, i] = 1
        elif ((z[i,j]==1)+(z[j,i]==1)): # at least one connection between i and j
            anatomy[i, j] = 2
            anatomy[j, i] = 2
        elif((np.sum(z[i]*z[:,j])+np.sum(z[:,i]*z[j]))): # one neuron connect i to j or j to i
            anatomy[i, j] = 3
            anatomy[j, i] = 3
        elif(np.sum(z[:,i]*z[:,j])): #at least one neuron project to both i, j
            anatomy[i, j] = 4
            anatomy[j, i] = 4
        else:                               # none of the above categories
            anatomy[i, j] = 5
            anatomy[j, i] = 5
np.savetxt('data/anatomy_categories_random.txt', anatomy, fmt='%d')
################################################################
