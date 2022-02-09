import os
from mpi4py import MPI
from time import sleep
################################################################
################################################################
################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
################################################################
stims = [400, 600, 800, 1000, 1200, 1400]
################################################################
'''
Batch simulation: each core will execute a simulations consisting in 10000 repetitions of the stimulus presentation
There will be 10 independent simulations per stimuli intensity, so the total of repetitions = 10^5.
#################################
Output files:
-) the spikes during simulation:
("Spikes_times.dat": spike neurons indx, "Spiking_neurons.dat": spike times relative to stimuli onset)
-) The mean conductance trajectory received per each neuron ("Cond.dat")
-) Each neuron times of the first spikes after stimulus presentation ("First_spk.dat")
-) Phase density distribution for each milisecond during the presentation of the stimuli ("pdens.dat")
#################################
Time control defined in main.cpp:
StepSize = 0.0001;
double tinit = -0.025, tend = 0.175; pre and post stimuli edges
double tmin = 0.6, tvar = 0.6;
double tadapt = tmin + rs.UniformRandom(0, tvar); Adaptation time between repetitions.
int nrep = 10000;
#################################
Simulations runs over compiled networks code located in folders
"psth_net": Small world network
"psth_ran": Randomly connected network
"psth_con": disconnected neurons
"psth_gcl": Disconnected network receiving barrage from a small-world network simulated in parallel
(i.e. resembling a conductance clamp experiment without feedback)

'''


folder = 'psth_gcl' ###
for k, proc in enumerate(range(60)):
    if (proc==rank):
        stim_indx = int(k/10)
        sim_indx = int(k%10)
        stim = stims[stim_indx]
        sleep(2.2*proc) # wait for different randomseed in c++ simulations
        os.mkdir('%s_%d_%d'%(folder, stim_indx, sim_indx))
        os.system('cp %s/main %s_%d_%d'%(folder, folder, stim_indx, sim_indx))
        os.chdir('%s_%d_%d'%(folder, stim_indx, sim_indx))
        os.system('./main %d'%stim) ## compiled network received stim intens as argument
        print(proc, '%s'%(folder))
##########################################################
##########################################################
##########################################################
