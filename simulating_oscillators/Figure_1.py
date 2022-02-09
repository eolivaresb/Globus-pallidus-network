import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate
################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
################################################################

from scipy.optimize import curve_fit
##################################################################
def func(x, c1, c2, c3, e1, e2, e3, p_thre):
    comp1 = (x**e1*(p_thre**e1 - x**e1))/(e1*p_thre**(1+2*e1)/(1 + 3*e1+2*e1**2))
    comp2 = (x**e2*(p_thre**e2 - x**e2))/(e2*p_thre**(1+2*e2)/(1 + 3*e2+2*e2**2))
    comp3 = (x**e3*(p_thre**e3 - x**e3))/(e3*p_thre**(1+2*e3)/(1 + 3*e3+2*e3**2))
    y = c1*comp1 + c2*comp2 + c3*comp3
    y[np.where(y<0)] = 0.0
    return y
##########################################################
N = 1000


plt.close('all')
fig = plt.figure(figsize = [7,7])
gm = gridspec.GridSpec(230, 140)

ax1 = plt.subplot(gm[50:95, :60])
ax2 = plt.subplot(gm[50:95, 80:], sharex= ax1)
ax31 = [plt.subplot(gm[120:145, 0:30]), plt.subplot(gm[150:180, 0:30]), plt.subplot(gm[185:225, 0:30])]
ax32 = [plt.subplot(gm[120:145, 35:65]), plt.subplot(gm[150:180, 35:65]), plt.subplot(gm[185:225, 35:65])]
ax33 = [plt.subplot(gm[120:145, 75:105]), plt.subplot(gm[150:180, 75:105]), plt.subplot(gm[185:225, 75:105])]
ax34 = [plt.subplot(gm[120:145, 110:140]), plt.subplot(gm[150:180, 110:140]), plt.subplot(gm[185:225, 110:140])]

##########################################################
##########################################################
ax0 = fig.add_axes([0.63, 0.82, 0.33, 0.16])

vt = np.loadtxt('vtrace.txt') ## voltage trace example
vt[0] -=vt[0][0]
vt[1] *= 1000
ti, ttot = 0.62, 0.205

ax0.plot(1000*(vt[0] - ti), vt[1], 'k')
ax0.set_xlim(0, 1000*ttot)
ax0.set_xlabel('Time (ms)')
ax0.set_ylabel('Voltage (mV)', labelpad = -2)

##########################################################
v = np.loadtxt('volt.dat').T  ## Mean voltage trajectory
ax1.plot(v[0], v[1]*1000, 'k')
ax1.set_ylim(-70, -50,)
ax1.set_xlim(0, 1)

##########################################################
prcs = np.loadtxt('prcs_data.txt')  ## experimental PRCs data
mean_prc = np.mean(prcs[:-1], axis = 0)
xphase_prc = prcs[-1]
popt, pcov = curve_fit(func, xphase_prc, mean_prc, p0 = [1, 1, 1, 1, 10, 30, 1], bounds=(0.0001, [50., 50., 50., 50., 50., 50., 1.0]) ) ## fit mean PRC using three compe=onent function

ax2.plot(np.linspace(0, 1, 1001), func(np.linspace(0, 1, 1001), *popt), ms = 0.9, color = 'k')

prc_func = interpolate.interp1d(np.append(np.append(0, xphase_prc), 1), np.append(np.append(0, xphase_prc), 0))
vfunc =  interpolate.interp1d(v[0], v[1])

#####################################
markers = 1
w = 30.46
dt = 0.0001
#####################################
## Simulate and plot phase model
#####################################
def plot_phase_cycle(Erev, amp, tstim, ax):
    indx_tstim = int(tstim/dt)
    t = np.arange(0, 0.15, dt)
    cond = np.zeros(len(t))
    curr = np.zeros(len(t))

    tau = 2.4/1000
    condwave = amp * np.exp(-np.arange(0, 0.1, dt)/tau)
    cond[indx_tstim:indx_tstim+len(condwave)] = condwave
    phase = np.zeros(len(t))
    phase2 = np.zeros(len(t))

    for i in range(len(t)-1):
        current = cond[i]*(Erev - vfunc(phase[i]))
        curr[i] = current
        phase[i+1] = phase[i] + dt*(w + current * prc_func(phase[i]))
        if phase[i+1] >= 1: phase[i+1] -=1
        phase2[i+1] = phase2[i] + dt*(w)
        if phase2[i+1] >= 1: phase2[i+1] -=1
    if markers:
        sp2 = np.where(np.diff(phase2) < -0.9)[0]
        phase2[sp2] = np.nan
        sp = np.where(np.diff(phase) < -0.9)[0]
        phase[sp] = np.nan
        ax[2].plot((t[sp])*1000, 1.04*np.ones(len(sp)), 'v', mec = 'k', ms = 4, mfc = 'k')
        ax[2].plot((t[sp2])*1000, 1.04*np.ones(len(sp2)), 'v', mec = 'k', ms = 4, mfc = 'w')

    ax[0].plot(t*1000, cond/1000, 'k')
    ax[1].plot(t*1000, curr, 'k')

    ax[2].plot(t*1000, phase, 'k')
    ax[2].plot(t*1000, phase2, '--k')
    ax[2].annotate('', xy=(1000*tstim,phase[indx_tstim]), xytext=(1000*tstim,1.1),
            arrowprops={'arrowstyle': '->','ls': 'dashed', 'lw': 1, 'color': 'b'})

    for a in ax:
        a.set_xlim(0, 37)


plot_phase_cycle(-0.073, 2500, 8/1000, ax31)
plot_phase_cycle(-0.073, 2500, 23/1000, ax32)
plot_phase_cycle(0, 1000, 8/1000, ax33)
plot_phase_cycle(0, 1000, 20/1000, ax34)

for ax in [ax0, ax1, ax2] + ax31 + ax32 + ax33 + ax34:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

for ax in [ax32, ax33, ax34]:
    for i in range(3):
        ax[i].spines['left'].set_visible(False)
        ax[i].set_yticks([])
for ax in [ax31, ax32, ax33, ax34]:
    ax[0].set_ylim(-0.12, 2.5)
    ax[1].set_ylim(-33, 95)
    ax[2].set_xlabel('Time (ms)')
    for i in range(2):
        ax[i].spines['bottom'].set_visible(False)
        ax[i].set_xticks([])

x1, x2, y= 0.25, 0.75, 0.47
plt.figtext(x1, y, '$E_{rev} = -73 mV$', {'color': 'k', 'fontsize': 9, 'ha': 'center', 'va': 'center', 'bbox': dict(boxstyle="round", fc="w", ec="w", pad=0.2)})

plt.figtext(x2, y, '$E_{rev} = 0 mV$', {'color': 'k', 'fontsize': 9, 'ha': 'center', 'va': 'center', 'bbox': dict(boxstyle="round", fc="w", ec="w", pad=0.2)})


ax2.set_ylabel('Cycles (pA s)'+'$^{-1}$')
for ax in [ax1, ax2]:
    ax.set_xlabel('Phase')

ax1.set_ylabel('Voltage (mV)', labelpad = 5)
ax31[0].set_ylabel('G (nS)', labelpad = 13)
ax31[1].set_ylabel('I (pA)', labelpad = 6)
ax31[2].set_ylabel('Phase', labelpad = 7)

x1, x2, y1, y2, y3, y4, fz = 0.02, 0.54, 0.98, 0.77, 0.5, 0.26, 16
plt.figtext(x1, y1, 'A', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x2, y1, 'B', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y2, 'C', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x2, y2, 'D', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y3, 'E', ha = 'center', va = 'center', fontsize = fz)
#plt.figtext(x2, y3, 'F', ha = 'center', va = 'center', fontsize = fz)

plt.subplots_adjust(left = 0.1, bottom = 0.05, right = 0.98, top = 0.94)
if markers:
    plt.savefig('figure1.png', dpi = 600)
    # plt.savefig('figure1.pdf')
else:
    plt.savefig('figure1v1.png', dpi = 600)
    # plt.savefig('figure1v1.pdf')
