import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import pandas as pd
##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
################################################################
def func(x, c1, c2, c3, e1, e2, e3, p_thre):
    comp1 = (x**e1*(p_thre**e1 - x**e1))/(e1*p_thre**(1+2*e1)/(1 + 3*e1+2*e1**2))
    comp2 = (x**e2*(p_thre**e2 - x**e2))/(e2*p_thre**(1+2*e2)/(1 + 3*e2+2*e2**2))
    comp3 = (x**e3*(p_thre**e3 - x**e3))/(e3*p_thre**(1+2*e3)/(1 + 3*e3+2*e3**2))
    y = c1*comp1 + c2*comp2 + c3*comp3
    y[np.where(y<0)] = 0.0
    return y
#################################################################
dt = 0.0001
ttot = 2 # 2 seconds saved phases, 1800 seconds total simulation
tsim = 1800
time = np.arange(0, ttot, dt)
#######################################
path_to_files = './../simulation_files'
volt = np.loadtxt('%s/volt.dat'%path_to_files)
prcfit = np.loadtxt('%s/PRCs/params_prc.txt'%path_to_files)
g = np.load('poisson_cond.npy', allow_pickle = 'true')[:20000]
################################################################
colors = ['#1b9e77','#d95f02','#7570b3']#['g', 'r', 'b']
labels = ['N1', 'N2', 'N3']
xphase = volt[:,0]
v = volt[:,1]
w = 30.46
n1, n2, n3 = 1, 10, 16
w1, w2, w3 = 20, 30, 40
neurons = [n1, n2, n3]
ws = [w1, w2, w3]
#################################################################
def cvisi(spk):
    s = np.diff(np.array(spk))
    return (np.std(s)/np.mean(s))
ytriang = 1.05 # y axis position of spike marker
#################################################################
################################################################
[phases, spikes, pdens] = np.load('data_prcs.npy', allow_pickle = 'True')
#################################################################
plt.close('all')
fig = plt.figure(figsize = [6.5,6.3])
gm = gridspec.GridSpec(200, 80, figure = fig)

ax3 = [plt.subplot(gm[:30,i*30:20+30*i]) for i in range(3)]

ax1 = plt.subplot(gm[45:63,:60])
axt = [plt.subplot(gm[65+30*i:91+i*30,:60]) for i in range(3)]
axpd = [plt.subplot(gm[65+30*i:91+i*30,61:80]) for i in range(3)]

ax2 = plt.subplot(gm[170:198,:20])
axisi = [plt.subplot(gm[170:193,30+i*16:44+i*16]) for i in range(3)]
#############   Plot conductance ########################################
tinit = 0.5
tfin = 0.394
ax1.plot(time-tinit, g/1000, 'k', lw = 1.4)
ax1.set_ylabel('G(t)', labelpad = 5)
############   Plot Phases ########################################
for i in range(3):
    pindx = np.copy(phases[i])
    sp = np.where(np.diff(pindx) < -0.9)[0]
    pindx[sp] = np.nan

    freerunning = (w*time-tinit)%1
    sfr = np.where(np.diff(freerunning) < -0.9)[0]
    freerunning[sfr] = np.nan
    axt[i].plot(time-tinit, freerunning, colors[i], linestyle = '--', alpha = 0.5, lw = 1)
    axt[i].plot(time[sfr]-tinit, ytriang*np.ones(len(sfr)), 'v', alpha = 0.5, color = colors[i], ms = 3, mfc = 'w')

    axt[i].plot(time-tinit, pindx, colors[i], alpha = 0.8)
    axt[i].plot(time[sp]-tinit, ytriang*np.ones(len(sp)), 'v', color = colors[i], ms = 3)

    axpd[i].fill_betweenx(np.linspace(0, 1, 1000), pdens[i]/(10*tsim), color = colors[i])
    axpd[i].set_yticklabels([])
    axt[i].set_ylabel('Phase')
    axpd[i].set_xlim(0, 2.8)
    for ax in [axpd[i], axt[i]]: ax.set_ylim(-0.01, 1.1)
    axpd[i].plot([1, 1], [0, 1], color = 'w', alpha = 0.5)
    axpd[i].plot([1, 1], [0, 1], '--', color = colors[i], alpha = 0.95)
    axpd[i].spines['left'].set_bounds(0, 1)
    axt[i].spines['left'].set_bounds(0, 1)
    axt[i].set_xticks([0, 0.10, 0.20, 0.30])

axt[2].set_xlabel('Time (ms)')
axpd[2].set_xlabel('Phase density')
axt[2].set_xticklabels(['%d'%(1000*d) for d in [0, 0.10, 0.20, 0.30]])

for ax in [ax1]+axt:
    ax.set_xlim(0, tfin)

for r in range(2):
    axt[r].set_xticklabels([])
    axpd[r].set_xticklabels([])

for ax in ax3 + [ax1, ax2] + axt + axpd + axisi:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xticks([])
##########  rate plots #####
for i in range(3):
    ax2.plot([0, 1], [w, len(spikes[i])/tsim], color = colors[i])
    print(i, colors[i], len(spikes[i])/tsim, (w-len(spikes[i])/tsim)/w)

ax2.set_ylim(15, 33)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['$\omega$', 'simulated'])
ax2.set_ylabel('Rate (Hz)')

##########  isi distributions #####
for i in range(3):
    h = np.histogram(np.diff(spikes[i]), bins = 250, range=[0.03, 0.125])
    axisi[i].fill_between(h[1][1:], h[0], color = colors[i], lw = 0)

for i, ax in enumerate(axisi):
    ax.text(0.7, 0.4, 'CV = %.2f'%cvisi(spikes[i]), ha='center', va='center', fontsize = 7.5, transform=ax.transAxes)
    ax.set_yticks([])
    ax.set_ylim(0, 1000)
for ax in axisi[1:]: ax.spines['left'].set_visible(False)
axisi[0].spines['left'].set_position(('outward', 3))
axisi[0].set_ylabel('Counts')
for ax in axisi:
    ax.spines['bottom'].set_position(('outward', 3))
    ax.set_xticks([0.05, 0.1])
    ax.set_xticklabels(['50','100'])
    ax.set_xlabel('ISI (ms)')
##########  PRC plots #####
for i , neu in enumerate(neurons):
    ax3[i].plot(xphase, func(xphase, *prcfit.T[neu]), color = colors[i], label = labels[i])
    ax3[i].set_ylabel('Cycles (pA s)'+'$^{-1}$', labelpad = 6)
    ax3[i].set_xlabel('Phase')
    ax3[i].set_ylim(-0.1, 2.8)
    ax3[i].set_xlim(0, 1)
    ax3[i].set_xticks([0, 0.5, 1])
################################################################
x1, x2, x3, x4 = 0.02, 0.335, 0.665, 0.39
y1, y2, y3, y4, y5, y6, fz = 0.98, 0.77, 0.67, 0.53, 0.39, 0.2, 13
plt.figtext(x1, y1, 'A1', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x2, y1, 'A2', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x3, y1, 'A3', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y2, 'B ', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y3, 'C1', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y4, 'C2', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y5, 'C3', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y6, 'D ', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x4, y6, 'E ', ha = 'center', va = 'center', fontsize = fz)
################################################################
fig.subplots_adjust(left = 0.11, bottom = 0.06, right = 0.98, top = 0.98)
plt.savefig('figure4.png', dpi = 300)
# plt.savefig('figure_4.pdf')
