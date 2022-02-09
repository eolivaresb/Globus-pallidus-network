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
dtbin = 0.5  # 4 ms
lb, ub = -25, 175
hrange = [lb, ub]
tbin = np.arange(lb, ub, dtbin)
############### load population histograms#########################
sims = [ 'con', 'net', 'ran', 'gcl']
sims_labels = ['Disconnected','Small world Network', 'Random Network', 'Barrage Only']
###########################################################
stim = 3  # position of 1000 pS in stimulus array
fz = 12
###########################################################
tau = 3.5 / 1000
tcond = np.arange(lb, ub, 0.0001)
condstim = 1.0 * np.exp(-tcond / tau)
condstim[np.where(tcond < 0)] = 0

hists = [np.load('data/hist_%s_%d.npy' %
                 (s, stim), allow_pickle=True) for s in sims]


plt.close('all')
fig = plt.figure(figsize=[10, 6])
gm = gridspec.GridSpec(132, 114)

ax1 = [plt.subplot(gm[4:12, i * 30:i * 30 + 24]) for i in range(4)]
ax2 = [plt.subplot(gm[20:52, i * 30:i * 30 + 24]) for i in range(4)]
ax3 = [plt.subplot(gm[60:92, i * 30:i * 30 + 24]) for i in range(4)]
ax4 = [plt.subplot(gm[100:132, i * 30:i * 30 + 24]) for i in range(4)]

n1, n2 = 114, 108
for i in range(4):
    ax1[i].plot(1000 * tcond, condstim, color='k')
    ax1[i].text(0.1, 1.7, sims_labels[i], color = 'k',
    transform=ax1[i].transAxes, fontsize=14)
    ### Population PSTH
    ax2[i].plot(tbin, np.mean(hists[i], axis = 0), 'k')
    ax2[i].set_ylim(0, 37)
    ### Neurons PSTH histograms
    ax3[i].plot(tbin, hists[i][n1], 'k')
    ax4[i].plot(tbin, hists[i][n2], 'k')
######################################
ax1[0].set_ylabel('nS', fontsize=fz)
for ax in ax1 + ax2 + ax3 + ax4:
    ax.set_xlim(-22, 123)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
for ax in ax3+ax4: ax.set_ylim(0, 52)
for ax in [ax2, ax3, ax4]:
    ax[0].set_ylabel('Spikes/s', fontsize=fz)
for ax in ax1:
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
for ax in ax4: ax.set_xlabel('Time (ms)', fontsize=fz)

#############################################################
plt.subplots_adjust(left=0.06, bottom=0.08, right=0.98, top=0.92)
plt.savefig('figure6.png', dpi=300)
plt.savefig('figure6.pdf')
