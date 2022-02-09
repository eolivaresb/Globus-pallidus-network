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
dat_con = [np.load('data/hist_analysis_con_%d.npy' %
                   s, allow_pickle=True).T for s in stims]
dat_net = [np.load('data/hist_analysis_net_%d.npy' %
                   s, allow_pickle=True).T for s in stims]
dat_gcl = [np.load('data/hist_analysis_gcl_%d.npy' %
                   s, allow_pickle=True).T for s in stims]

psth_disc = np.load('data/hist_con_3.npy', allow_pickle=True)

########################################
########################################


def reg(x, y):
    x = 1000. / x
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope


# Analysis only on neurons faster than 5 Hz
val = np.where((dat_net[3][base] > 5) * (dat_gcl[3][base] > 5))
# Separate neurons according to primary response lenght under barrage on stimuli of 1 nS
set1, set2 = np.where(dat_gcl[3][Tprim][val] > 10), np.where(
    dat_gcl[3][Tprim][val] < 10)

# calculate slope on linear regression ISI v/s Primary response width
slp_con, slp_gcl = np.zeros(lstims), np.zeros(lstims)
slp_ran1, slp_ran2 = np.zeros(lstims), np.zeros(lstims)
for sindx, stim in enumerate(stims):
    # Slope on disconnectd neurons and under barrage
    slp_con[sindx] = reg(dat_con[sindx][base], dat_con[sindx][Tprim])
    slp_gcl[sindx] = reg(dat_gcl[sindx][base][val], dat_gcl[sindx][Tprim][val])
    # Slope in connected neurons for the two subsets
    slp_ran1[sindx] = reg(dat_net[sindx][base][val][set1],
                          dat_net[sindx][Tprim][val][set1])
    slp_ran2[sindx] = reg(dat_net[sindx][base][val][set2],
                          dat_net[sindx][Tprim][val][set2])
#######################
xint = np.linspace(10, 200, 10)  # x value for isi interpolation
st = 3  # 1 nS stimuli
#######################
plt.close('all')
fig = plt.figure(figsize=[7.5, 7])
gm = gridspec.GridSpec(280, 100)
[ax1, ax2, ax3] = [plt.subplot(gm[i * 100:i * 100 + 80, :44])
                   for i in range(3)]
[ax4, ax5, ax6] = [plt.subplot(gm[i * 100:i * 100 + 80, 56:])
                   for i in range(3)]

# Panel A disconnected histograms
n1, n2, n3 = 13, 29, 119
x = np.repeat(np.arange(-25, 175.5, 0.5), 2)[1:-1]  # x axis time for PSTH
ax1.plot(x, np.repeat(psth_disc[n1], 2), 'k')
ax1.plot(x, np.repeat(psth_disc[n2], 2), 'g')
ax1.plot(x, np.repeat(psth_disc[n3], 2), 'orange')
ax1.set_xlim(-10, 26)
ax1.set_ylim(0, 65)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Spikes/s')
for [ypos, str] in [[53, '1'], [29, '2'], [19, '3']]:
    ax1.text(-4, ypos, str, fontsize=11)

for [xpos, ypos, str] in [[9, 61, '20'], [7, 36, '40'], [13, 12, '60']]:
    ax1.text(xpos, ypos, 'ISI = ' + str + ' ms', fontsize=11)

# Panel B ISI v/s primary response width disconnected
x, y = 1000. / dat_con[st][base],  dat_con[st][Tprim]
ax4.plot(x, y, '.', color='0.3', ms=1.6, alpha=0.5)
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax4.plot(xint, slope * xint + intercept, '-', color='k')

for n, c, t in zip([n1, n2, n3], ['k', 'g', 'orange'], ['1', '2', '3']):
    ax4.plot([1000. / dat_con[st][base][n]],
             [dat_con[st][Tprim][n]], '.', color=c, ms=8)
    ax4.text(1000. / dat_con[st][base][n], 1.2 +
             dat_con[st][Tprim][n], t, fontsize=11)
ax4.set_xlim(19, 68)
ax4.set_ylim(0, 12)
ax4.set_xlabel('ISI (ms)')
ax4.set_ylabel(r'$1^{o}$' + ' Resp Width (ms)')

# Panel C ISI v/s primary response width disconnected & Barrage
ax2.plot(1000. / dat_con[st][base], dat_con[st][Tprim], '.',
         color='0.3', ms=1.6, alpha=0.5, label='Disconnected')
ax2.plot(1000. / dat_gcl[st][base][val], dat_gcl[st][Tprim][val],
         '.', color='orange', ms=1.6, alpha=0.5, label='Barrage')
leg2 = ax2.legend(markerscale=4)
for lh in leg2.legendHandles: lh._legmarker.set_alpha(1)
ax2.set_xlabel('ISI (ms)')
ax2.set_ylabel(r'$1^{o}$' + ' Resp Width (ms)')

# Panel E ISI v/s primary response width connected Network
x, y = 1000. / dat_net[st][base][val][set1], dat_net[st][Tprim][val][set1]
ax3.plot(x, y, '.', color='b', ms=1.6, alpha=0.5,
         label='Barrage ' + r'$1^{o}$' + ' Width > 10ms')
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax3.plot(xint, slope * xint + intercept, '--', color='b')

x, y = 1000. / dat_net[st][base][val][set2], dat_net[st][Tprim][val][set2]
ax3.plot(x, y, '.', color='r', ms=1.6, alpha=0.5,
         label='Barrage ' + r'$1^{o}$' + ' Width < 10ms')
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
ax3.plot(xint, slope * xint + intercept, '--', color='r')

leg3 = ax3.legend(markerscale=4)
for lh in leg3.legendHandles: lh._legmarker.set_alpha(1)
ax3.set_xlabel('ISI (ms)')
ax3.set_ylabel(r'$1^{o}$' + ' Resp Width (ms)')


# Panel D Slope ISI v/s primary response width disconnected & Barrage
ax5.plot(0.001 * np.array(stims), slp_con, '.k', label='Disconnected')
ax5.plot(0.001 * np.array(stims), slp_gcl,
         '.', color='orange', label='Barrage')

# Panel F Slope ISI v/s primary response width connected Network
ax6.plot(0.001 * np.array(stims), slp_ran1, '.b',
         label='Barrage ' + r'$1^{o}$' + ' Width > 10ms')
ax6.plot(0.001 * np.array(stims), slp_ran2, '.r',
         label='Barrage ' + r'$1^{o}$' + ' Width < 10ms')
for ax in [ax5, ax6]:
    ax.legend()
    ax.set_ylabel(r'$1^{o}$' + ' Resp Fractional Width')
    ax.set_xlabel('Stimulus peak G (nS)')
    ax.set_ylim(0, 0.18)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.99)
plt.savefig('./figure7.png', dpi=300)
plt.savefig('figure7.pdf')
