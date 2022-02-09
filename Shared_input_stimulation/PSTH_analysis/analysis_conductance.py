import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
##########################################################
N = 1000
dtbin = 0.1  # 0.1 ms timestep on simulation
lb, ub = -25, 175
hrange = [lb, ub]
time = np.arange(lb, ub, dtbin)
bins = len(time)
ppre = np.where(time <= 0)[0][-1]  # time bins before pulse
############### load population histograms################
sims = ['net', 'con', 'ran', 'gcl']
stimulis = [400, 600, 800, 1000, 1200, 1400]

###########################################################
##############         PSTH analysis          #############
###########################################################
'''
Data structure
N = 1000 rows, one per neuron
--- 8 columns with psth analysis:
0: 'baseline',  Baseline rate
1: 'peak',      Peak primary response
2: 'area',      Area of primary response
3: 'Tprim',     Width of primary response (time crossing above baseline)
4: 'tpeak',     Time for peak in the primary response
5: 'hwidth',    Half width rpimary response
6: 'postpeak'   Peak secondary response
7: 'tpostpeak'  Time for peak in secondary response
'''

for sim in ['net', 'ran']:
   for k, stim in enumerate(stimulis):
       data = np.zeros((N, 8))
       cond = np.load('data/cond_%s_%d.npy' % (sim, k), allow_pickle=True)
       for n in range(N):
           hf = cond[n]  # conductance trace
           baseline = np.mean(hf[1:ppre]) #first point got a zero
           # get cummulative conductance in pS*msec with respect to baseline
           cs = dtbin * np.cumsum(hf - baseline)
           ########################################
           #########  Primary deshinibition  ######
           peak = np.min(hf[ppre:])  # minimum prim response
           # position of crossing baseline up
           # endfirstresp = np.argmin(cs)
           endfirstresp = ppre + 19 + np.where(hf[ppre + 20:] >baseline)[0][0]
           area = cs[endfirstresp]  # area from cummulative deshinibition
           Tprim = time[endfirstresp]  # Time of crossing baseline
           # time of primary response
           tpeak = time[ppre + 19 + np.argmin(hf[ppre + 20:])]
           # half width time
           mp = (baseline + peak) / 2  # midpoint between peak and baseline
           hp0 = np.where(hf[1:] < mp)[0][0]  # first time below mp
           hp1 = np.where(hf[1:] < mp)[0][-1]  # last time below mp
           hwidth = (time[hp1] - time[hp0])
           ########################################
           ###########  Secondary overshoot #######
           # peak secondary response
           postpeak = np.max(hf[ppre + 1:])
           # seondary response time
           tpostpeak = time[ppre + 19 + np.argmax(hf[ppre + 20:])]
           ########################################
           ############     save data       #######
           data[n] = [baseline, peak-baseline, area, Tprim,
                      tpeak, hwidth, postpeak-baseline, tpostpeak]
       np.save('data/cond_analysis_%s_%d.npy'%(sim, stim), data)
       # np.savetxt('data_text_files/cond_analysis_%s_%d.dat'%(sim, stim), data)

basal_cond_ran = np.load('data/cond_analysis_ran_%d.npy'%(1000), allow_pickle=True)[:,0]
basal_cond_net = np.load('data/cond_analysis_net_%d.npy'%(1000), allow_pickle=True)[:,0]

plt.close('all')
fig = plt.figure(figsize = [8.3,3.6])
gm = gridspec.GridSpec(100, 100)
ax1 = plt.subplot(gm[:, :44])
ax2 = plt.subplot(gm[:, 56:])
###############
ax1.hist(basal_cond_net/1000, bins = 60, range = [0.7, 1.7], color = 'k')
ax1.text(0.1, 1.1, 'Small_world Network\n Mean G in network = %.2f nS'%np.mean(basal_cond_net/1000), transform=ax1.transAxes)
###############
ax2.hist(basal_cond_ran/1000, bins = 60, range = [0.3, 2.7], color = 'k')
ax2.text(0.1, 1.1, 'Random Network\n Mean G in network = %.2f nS'%np.mean(basal_cond_ran/1000), transform=ax2.transAxes)
###############
for a in [ax1, ax2]:
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.set_xlabel('mean G (nS)')
    a.set_ylabel('Counts')
#    a.set_xlim(0.69, 1.71)
####################
fig.subplots_adjust(left = 0.11, bottom = 0.12, right = 0.98, top = 0.8)
plt.savefig('Mean_conductances.png', dpi = 300)
