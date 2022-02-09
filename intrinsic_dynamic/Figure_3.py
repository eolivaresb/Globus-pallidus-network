import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import os
import pandas as pd
#plt.rcParams["image.composite_image"] =False
################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
################################################################
N = 1000
ttot = 1800
################################################################
################################################################
plt.close('all')
fig = plt.figure(figsize = [7,5])
gm = gridspec.GridSpec(70, 105, figure = fig)
axz1 = plt.subplot(gm[3:22, 0:23])
axz2 = plt.subplot(gm[3:22, 60:83])
axz = [axz1, axz2]
axz1z = plt.subplot(gm[:12, 28:44])
axz2z = plt.subplot(gm[:12, 88:104])
axzz = [axz1z, axz2z]
axz1h = plt.subplot(gm[14:22, 30:45])
axz2h = plt.subplot(gm[14:22, 90:105])
axzh = [axz1h, axz2h]
ax1 = [plt.subplot(gm[33:46, :45]), plt.subplot(gm[33:46,60:])]
ax2 = [plt.subplot(gm[54:70, :45]), plt.subplot(gm[54:70,60:])]
###############################################################
net_names = ['Small-world', 'Random']
for col, net in enumerate(['small', 'random']):
    ###############################################################
    z =  np.loadtxt('./../simulation_files/network_matrix_%s.dat'%net).astype(int)
    zinvert = np.abs(z-1)
    [rates, cv] = np.loadtxt('./%s/rates_and_cv.dat'%net)
    w = np.loadtxt('./../simulation_files/neurons.dat')[:,0]
    ################## plot network architecture  ###################################
    ##########################################################
    axz[col].imshow(zinvert, cmap=plt.cm.gray, origin = 'lower', interpolation='None')
    axz[col].plot([949, 949, 999, 999, 949], [999, 949, 949, 999, 999], 'r', lw = 0.6)
    axzz[col].imshow(zinvert[-50:, -50:], cmap=plt.cm.gray, origin = 'lower', interpolation='None')
    axzh[col].hist(np.sum(z, axis = 1), color = 'k', bins = np.arange(24))

    ################## plot rates and cvs  ###################################
    rbins = 50
    hw = np.histogram(w, bins = rbins, range = [0, 52])
    ax1[col].plot(np.repeat(hw[1], 2)[1:-1], np.repeat(hw[0], 2), 'g', alpha = 0.85)
    h = np.histogram(rates, bins = rbins, range = [0, 52])
    ax1[col].plot(np.repeat(h[1], 2)[1:-1], np.repeat(h[0], 2), 'k', alpha = 0.85)
    h2 = np.histogram(cv[np.where(rates>2)], bins = 28, range = [0, 0.8])
    ax2[col].plot(np.repeat(h2[1], 2)[1:-1], np.repeat(h2[0], 2), 'k', alpha = 0.85)
    ax1[col].set_xlabel('Rate (Hz)')
    ax1[col].set_ylabel('Counts')
    ax2[col].set_xlabel('CV isis')
    ax2[col].set_ylabel('Counts')

###################  Spike count correlation #########################
###################################################
for k, ax in enumerate(ax1+ ax2+axzh):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position(('outward', 1))
    ax.spines['bottom'].set_position(('outward', 1))

for ax in ax1:
    ax.set_xlim(-0.02, 52)
    ax.set_ylim(-1, 75)

for ax in ax2:
    ax.set_xlim(-0.01, 1.03)
    ax.set_ylim(-2, 208)
    ax.set_xticks([0.0, 0.5, 1])
    ax.set_xticklabels(['0','0.5', '1'])

for ax in axz+axzz+axzh:
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.43)

for ax in axzz:
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_color('red')

for ax in axzh:
    ax.set_yticks([])
    ax.set_xlim(0, 20)
    ax.set_xticks([0.5, 10.5, 20.5])
    ax.set_ylabel('Counts', fontsize = 7)
    ax.set_xticklabels(['0','10', '20'], fontsize = 7)
    ax.set_xlabel('Synaptic inputs', fontsize = 7)

for ax in axz:
    ax.set_xticks([0, 999])
    ax.set_xticklabels(['1', 'N'], fontsize = 9)
    ax.set_yticks([0, 999])
    ax.set_yticklabels(['1', 'N'], fontsize = 9)
    ax.set_xlabel('Postsynaptic\nneuron', labelpad = -4)
    ax.set_ylabel('Presynaptic\nneuron', labelpad = -4)

for ax in axzz:
    ax.set_xticks([])
    ax.set_yticks([])

plt.figtext(0.12, 0.97, 'Small-world', fontsize = 14, ha = 'left')
plt.figtext(0.63, 0.97, 'Random', fontsize = 14, ha = 'left')

##################################################################
x1, x2, y1, y2, y3, fz = 0.025, 0.525, 0.97, 0.58, 0.32, 16
plt.figtext(x1, y1, 'A', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x2, y1, 'B', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y2, 'C', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x2, y2, 'D', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y3, 'E', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x2, y3, 'F', ha = 'center', va = 'center', fontsize = fz)
################################################################
fig.subplots_adjust(left = 0.1, bottom = 0.09, right = 0.98, top = 0.98)
plt.savefig('figure3.png', dpi = 300)
# plt.savefig('figure3.pdf')
