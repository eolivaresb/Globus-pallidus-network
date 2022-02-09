import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
################################################################
tbin = 10/1000 # 10 miliseconds
N = 1000
ttot = 1800

def cv(x):
    return (np.std(x)/np.mean(x))
    ###################################
    
'''
Here's to check the variability in rate during the 1800 s of simulation
there will be 180 rates stimations for windows of 10 seconds each
the CV of rates is going to be calculated for each neuron and plotted agains it mean rate
rate variability is similar in both network architectures. 
rate CV is below 0.1 for all neurons that fires above 5 Hz
and below 0.05 for those firing above 15 Hz
'''
w = 10
################################################################
data = np.array(pd.read_csv('small_10/Spikes.dat', sep='\s+', header=None))
spikes = [data[np.where(data[:,1] == i), 0][0] for i in range(1, N+1)]

cv_rates = np.zeros(N)
rates = np.zeros((2, N))
fr = np.array([len(s)/ttot for s in spikes])
for i in range(N):
    s = spikes[i]
    r = np.array([len(np.where((s>i*w)*(s<(i+1)*w))[0])/w for i in range((int(1800/w)))])
    cv_rates[i] = cv(r)
    rates_edges[i] = [np.min(r), np.max(r)]
plt.plot(fr, cv_rates, '.r')
################################################################
data = np.array(pd.read_csv('random_10/Spikes.dat', sep='\s+', header=None))
spikes = [data[np.where(data[:,1] == i), 0][0] for i in range(1, N+1)]

cv_rates = np.zeros(N)
fr = np.array([len(s)/ttot for s in spikes])
for i in range(N):
    s = spikes[i]
    r = np.array([len(np.where((s>i*w)*(s<(i+1)*w))[0])/w for i in range((int(1800/w)))])
    cv_rates[i] = cv(r)
    rates_edges[i] = [np.min(r), np.max(r)]

plt.plot(fr, cv_rates, '.b')

plt.show()
################################################################

