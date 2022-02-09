import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
################################################################
N = 1000
ttot = 1800
twindow, tbin = 0.5, 0.001
timecc = np.arange(-twindow, twindow+tbin, tbin)
################################################################
anatomy_categories = ['Self', u'\u24D8\u2194\u24D9',u'\u24D8\u2192\u24D9', u'\u24D8\u2192\u24de\u2192\u24D9', u'\u24D8\u2190\u24de\u2192\u24D9', 'Other']
colors = ['k', 'g']
flierprop = dict(markerfacecolor='k', marker='.', markersize = 0.5)
boxprops = dict(linestyle='-', linewidth=0.6, color='k')
################################################################
plt.close('all')
fig = plt.figure(figsize = [6,6])
gm = gridspec.GridSpec(160, 100, figure = fig)
axpear = [plt.subplot(gm[2:35, 60*i:40+60*i]) for i in range(2)]
axccorr = [[plt.subplot(gm[50+22*j:50+20+22*j, 60*i:40+60*i]) for j in range(5)] for i in range(2)]
###############################################################
#
net_names = ['Small-world', 'Random']
for col, net in enumerate(['small', 'random']):
    ###############################################################
    ValidCC = np.loadtxt('crosscorrelations/data/ValidCC_%s'%net).astype(int)
    z =  np.loadtxt('./../simulation_files/network_matrix_%s.dat'%net).astype(int)
    [rates, cv] = np.loadtxt('./%s/rates_and_cv.dat'%net)
    pearson_corr = np.load('./%s/corr.npy'%net, allow_pickle = 'true')
    [mean_cc, npairs] = np.load('crosscorrelations/mean_cc_%s.npy'%net, allow_pickle = 'true')
    anatomy = np.loadtxt('crosscorrelations/data/anatomy_categories_%s.txt'%net).astype(int)
##########################################################
###################  Spike count correlation #########################
    nbins = 100
    ax = axpear[col]
    all_data = [pearson_corr[np.where((anatomy==k+1)*(ValidCC==1))] for k in range(5)]
    print('mean pearson %s'%net, [np.mean(a) for a in all_data])
    print('std pearson %s'%net, [np.std(a) for a in all_data])

    bplot = ax.boxplot(all_data, notch=True, vert=True, patch_artist=True, labels=anatomy_categories[1:], flierprops=flierprop, boxprops=boxprops)
    for patch in bplot['boxes']: patch.set_facecolor('0.8')
    ax.axhline(y=0, linestyle='--', color = 'k', linewidth = 0.4)

    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(9)
                tick.label.set_rotation(20)
#    ax.set_xlabel('Connectivity')
    ax.set_ylabel('Spike count\ncorrelation')
    ax.set_ylim(-0.33, 0.24)
    ax.set_yticks([-0.2, 0, 0.2])
###################  Spike mean crosscorrelation #########################
###################################################
    bases = [[6, 6], [6, 6], [17., 17.], [18, 18], [18.415, 19.59]]
    ticks = [[[10, 20], [10, 20], [18, 20], [18, 19], [18.42, 18.45]], [[10, 20], [10, 20], [18, 20], [18, 19], [19.60, 19.63]]]
    span = [20 , 24, 4.0, 1.6, 0.05]
    ytext = [0.2, 0.2, 0.6, 0.7, 0.77]
    for k, categ in enumerate(anatomy_categories[1:]):
        ax = axccorr[col][k]
        ax.plot(1000*timecc, mean_cc[k]/npairs[k], color = 'k', label = anatomy_categories[k+1])
        ax.text(0.05, ytext[k], anatomy_categories[k+1], ha = 'left', va = 'bottom', fontsize = 12, transform=ax.transAxes)
        ax.text(0.95, ytext[k], '%d pairs'%npairs[k], ha = 'right', va = 'bottom', fontsize = 9, transform=ax.transAxes)
        ax.set_ylim(bases[k][col], bases[k][col]+span[k])
        ax.set_yticks(ticks[col][k])
###################################################
for k, ax in enumerate(axpear):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

pads = [12, 11, 10, 11, 0, 12, 12, 10, 10, 0]
for k, ax in enumerate(axccorr[0]+axccorr[1]):
    ax.axvline(x = 0, linestyle = '--', linewidth = 0.2, color = 'k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_xlim(-100, 100)
    ax.set_ylabel('Rate (Hz)', labelpad = pads[k], fontsize = 9)

for k, ax in enumerate(axccorr[0][:-1]+axccorr[1][:-1]):
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])

for ax in [axccorr[0][4], axccorr[1][4]]:
    ax.set_xticklabels(['%d'%d for d in [-100, -50, 0, 50, 100]])
    ax.set_xlabel('Time lag (ms)')

plt.figtext(0.13, 0.97, 'Small-world', fontsize = 15, ha = 'left')
plt.figtext(0.64, 0.97, 'Random', fontsize = 15, ha = 'left')

##################################################################
x1, x2, y1, y2, fz = 0.025, 0.525, 0.96, 0.7, 16
plt.figtext(x1, y1, 'A', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x2, y1, 'B', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y2, 'C', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x2, y2, 'D', ha = 'center', va = 'center', fontsize = fz)
################################################################
fig.subplots_adjust(left = 0.12, bottom = 0.06, right = 0.98, top = 0.96)
plt.savefig('figure5.png', dpi = 300)
# plt.savefig('figure5.pdf')
