import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy import stats
################################################################
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update(
    {'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix', })
###############################################################
##########################################################
###########################################################
stims = [400, 600, 800, 1000, 1200, 1400]
lstims = len(stims)
###########################################################
base, peak, area, Tprim, tpeak, hwidth, secpeak, tsecpeak = 0, 1, 2, 3, 4, 5, 6, 7
# '''
# Data structure
# N = 1000 rows, one per neuron
# --- 8 columns with psth analysis:
# 0: 'baseline',  Baseline rate
# 1: 'peak',      Peak primary response
# 2: 'area',      Area of primary response
# 3: 'Tprim',     Width of primary response (time crossing above baseline)
# 4: 'tpeak',     Time for peak in the primary response
# 5: 'hwidth',    Half width rpimary response
# 6: 'postpeak'   Peak secondary response
# 7: 'tpostpeak'  Time for peak in secondary response
# '''
###########################################################
# Load data
dat_net = [np.load('data/hist_analysis_net_%d.npy' %
                   s, allow_pickle=True).T for s in stims]
dat_gcl = [np.load('data/hist_analysis_gcl_%d.npy' %
                   s, allow_pickle=True).T for s in stims]

psth_net = np.load('data/hist_net_3.npy', allow_pickle=True)
psth_gcl = np.load('data/hist_gcl_3.npy', allow_pickle=True)

########################################
########################################
# Analysis only on neurons faster than 5 Hz
val = np.where((dat_net[3][base] > 5) * (dat_gcl[3][base] > 5))
# Separate neurons according to primary response lenght under barrage on stimuli of 1 nS
set1, set2 = np.where(dat_gcl[3][Tprim][val] > 10), np.where(
    dat_gcl[3][Tprim][val] < 10)
#######################
# calculate mean rebound for the two neurons subsets
msec_ran1, msec_ran2 = np.zeros(lstims), np.zeros(lstims)
for sindx, stim in enumerate(stims):
    msec_ran2[sindx] = np.mean(dat_net[sindx][secpeak][val][set2])
    msec_ran1[sindx] = np.mean(dat_net[sindx][secpeak][val][set1])
#######################
########################################

st = 3  # 1 nS stimuli
#######################
plt.close('all')
fig = plt.figure(figsize=[8, 5.5])
gm = gridspec.GridSpec(180, 100)
[ax1, ax2] = [plt.subplot(gm[i * 100:i * 100 + 80, :44])
              for i in range(2)]
[ax3, ax4] = [plt.subplot(gm[i * 100:i * 100 + 80, 56:])
              for i in range(2)]

n1, n2 = 754, 442
# Panel A PSTH for two example neurons in the connected network
x = np.repeat(np.arange(-25, 175.5, 0.5), 2)[1:-1]  # x axis time for PSTH
ax1.plot(x, np.repeat(psth_net[n1], 2), 'r')
ax1.plot(x, np.repeat(psth_net[n2], 2), 'b')
for [xpos, ypos, str] in [[9, 28, '1'], [8, 16, '2']]:
    ax1.text(xpos, ypos, str, fontsize=11)

# Panel B PSTH for same neurons under barrage only
x = np.repeat(np.arange(-25, 175.5, 0.5), 2)[1:-1]  # x axis time for PSTH
ax2.plot(x, np.repeat(psth_gcl[n1], 2), 'r')
ax2.plot(x, np.repeat(psth_gcl[n2], 2), 'b')
for [xpos, ypos, str] in [[9, 23, '1'], [10, 15, '2']]:
    ax2.text(xpos, ypos, str, fontsize=11)
######################
for ax in [ax1, ax2]:
    ax.set_xlim(-10, 26)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Spikes/s')


# Panel C Secondary response in two grupos of neurons in Connected Network

ax3.plot(dat_net[st][secpeak][val][set1], dat_net[st][tsecpeak][val]
         [set1], '.b', ms=1.6, label='Barrage ' + r'$1^{o}$' + ' Width > 10ms')
ax3.plot(dat_net[st][secpeak][val][set2], dat_net[st][tsecpeak][val]
         [set2], '.r', ms=1.6, label='Barrage ' + r'$1^{o}$' + ' Width < 10ms')

arrowprops = dict(arrowstyle = "->",)
x, y = dat_net[st][secpeak][n1], dat_net[st][tsecpeak][n1]
ax3.annotate('1', xy =(x, y), xytext =(x+0.5, y+15), arrowprops = arrowprops)
x, y = dat_net[st][secpeak][n2], dat_net[st][tsecpeak][n2]
ax3.annotate('2', xy =(x, y), xytext =(x+2, y+1.8), arrowprops = arrowprops)

ax3.set_xlabel(r'$2^{o}$' + ' Response Peak Ampl (spk/s)')
ax3.set_ylabel(r'$2^{o}$' + ' Response TTP')
ax3.legend( markerscale=4)


# Panel D Mean secondary response amplitude in two grupos of neurons in Connected Network
ax4.plot(0.001 * np.array(stims), msec_ran1, '.b',
         label='Barrage ' + r'$1^{o}$' + ' Width > 10ms')
ax4.plot(0.001 * np.array(stims), msec_ran2, '.r',
         label='Barrage ' + r'$1^{o}$' + ' Width > 10ms')

ax4.set_xlabel('Stimulus peak G (nS)')
ax4.set_ylabel(r'$2^{o}$' + ' Response Peak Ampl (spk/s)')
ax4.legend()


for ax in [ax1, ax2, ax3, ax4]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.99)
plt.savefig('./figure8.png', dpi=300)
plt.savefig('./figure8.pdf')
