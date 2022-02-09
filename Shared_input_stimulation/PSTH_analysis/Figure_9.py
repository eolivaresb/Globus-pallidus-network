import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
################################################################
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update(
    {'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix', })
###############################################################
##########################################################
dtbin = 0.0005  # 4 ms
lb, ub = -0.025, 0.175
hrange = [lb, ub]
tbin = np.arange(lb, ub, dtbin)# time axis for PSTHs
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
####        ###########################
tau = 3.5 / 1000
tcond = np.arange(lb, ub, 0.0001)
time = np.linspace(-25, 175, 2000)  # time axis for conductance traces
condstim = 1000 * np.exp(-tcond / tau)
condstim[np.where(tcond < 0)] = 0
###########################################################
#### Load data
###### Histograms
hnet = np.load('data/hist_net_3.npy', allow_pickle=True)
hgcl = np.load('data/hist_gcl_3.npy', allow_pickle=True)
#################  Conductances
gnet = np.load('data/cond_net_3.npy', allow_pickle=True)
################ Analysis histogram
hnet_dat = np.load('data/hist_analysis_net_1000.npy', allow_pickle=True).T
hgcl_dat = np.load('data/hist_analysis_gcl_1000.npy', allow_pickle=True).T
################ Analysis conductances
gnet_dat = np.load('data/cond_analysis_net_1000.npy', allow_pickle=True).T
###########################################################
def plot_hist(ax, x, y, c, label = ''):
    dx = x[1] - x[0]
    x = np.repeat(np.append(x, x[-1] + dx) - dx / 2., 2)[1:-1]
    ax.plot(x, np.repeat(y, 2), color=c, label = label)
# ###########################################################
# ###########################################################
fz = 12
plt.close('all')
fig = plt.figure(figsize=[7, 9.4])
gm = gridspec.GridSpec(220, 114)
axg = [plt.subplot(gm[5:50, :]), plt.subplot(gm[120:165, :])]
axh = [plt.subplot(gm[60:105, :]), plt.subplot(gm[175:220, :])]

for k, n in enumerate([442, 754]):
    ###########################################################
    ###########################################################
    axg[k].axhline(y = gnet_dat[base, n]/1000., linestyle = '--', color = 'k', linewidth = 0.5)
    axg[k].plot(time, (gnet_dat[base, n] + condstim)/1000., color='k',
                 lw=1.2, alpha=0.75, label='Total G(t) Barrage')
    axg[k].plot(time, gnet[n]/1000., '--b', label='Local G(t) Small-world network')
    axg[k].plot(time, (gnet[n] + condstim)/1000., color='b', lw=1.22, label='Total G(t) Small-world network')
    axg[k].legend()
    ###########################################################
    axh[k].axhline(y = hnet_dat[base, n], linestyle = '--', color = 'b', linewidth = 0.5)
    plot_hist(axh[k], 1000 * (tbin + dtbin / 2), hnet[n], 'b', 'PSTH Small-world')
    ###########################################################
    axh[k].axhline(y = hgcl_dat[base, n], linestyle = '--', color = 'k', linewidth = 0.5)
    plot_hist(axh[k], 1000 * (tbin + dtbin / 2), hgcl[n], 'k', 'PSTH Barrage')
    axh[k].legend()
    ###########################################################
for ax in axh:
    ax.set_xlabel('Time (ms)', fontsize=fz)
    ax.set_ylabel('Rate (spk/s)', fontsize=fz)
for ax in axg:
    ax.set_xticklabels([])
    ax.set_ylabel('G (nS)', fontsize=fz)
for ax in axg+ axh:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(-5, 25)
    ###########################################################
plt.subplots_adjust(left=0.1, bottom=0.08, right=0.98, top=0.99)
plt.savefig('./figure9.png', dpi=300)
plt.savefig('figure9.pdf')
