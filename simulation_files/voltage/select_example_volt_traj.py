import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

## Data was taken from neuron #14 m191119.2.2.hdf
## which has a firing rate near population mean rate
## It have a regular firing with almost no IPSC in sweep 1 between times 0.62 and 0.87 s

### loop to cheack sweeps
#for i in range(20):
#    v = np.loadtxt('VoltageDataForPRC/volt_%d.txt'%(i+1)).T
#    plt.plot(v[0], v[1]) 
#    plt.title(' Data %d'%(i+1))
#    plt.xlim(0, 0.25)
#    plt.show()

v = np.loadtxt('VoltageDataForPRC/volt_1.txt').T
np.savetxt('vtrace.txt', v)
np.savetxt('../../simulating_oscillators/vtrace.txt', v)
