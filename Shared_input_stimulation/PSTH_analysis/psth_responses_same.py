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
N = 1000
############### load population histograms#########################
sims = ['net', 'ran', 'con', 'gcl', 'gcl_ran']
sims_labels = ['Small world', 'Random', 'Disc', 'Barr SW', 'Barr Rand']
colors = ['k', 'k', 'r', 'g']
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
network = 'Random' #'Small_world'#

dat_con = [np.loadtxt('data/hist_analysis_con_%d.dat'%s).T for s in stims]
if (network == 'Small_world'):
    dat_ran = [np.loadtxt('data/hist_analysis_net_%d.dat'%s).T for s in stims]
    dat_gcl = [np.loadtxt('data/hist_analysis_gcl_%d.dat'%s).T for s in stims]
if (network == 'Random'):
    dat_ran = [np.loadtxt('data/hist_analysis_ran_%d.dat'%s).T for s in stims]
    dat_gcl = [np.loadtxt('data/hist_analysis_gcl_ran_%d.dat'%s).T for s in stims]

########################################
def plot_reg(ax, x, y, color, xtext):
    x = 1000./x
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    ax.plot(x, y, '.', color = color, ms = 1.2, alpha = 0.5)
    xint = np.linspace(10, 200, 10)
    ax.plot(xint, slope*xint + intercept, '--', color = color)
    ax.text(xtext, 0.92, 'slope = %.2f'%slope, transform=ax.transAxes, color = color)
    return slope

def reg(x, y):
    x = 1000./x
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return slope

def sec_resp(ax, x, y, color):
    ax.plot(x, y, '.', color = color, ms = 1.2, alpha = 0.5)
    ax.axvline(x = np.mean(x), color = color, linestyle = '--', linewidth = 1.2)
    return np.mean(x)

########################################


slp_con, slp_gcl = np.zeros(lstims), np.zeros(lstims)
slp_ran1, slp_ran2 = np.zeros(lstims), np.zeros(lstims)
msec_ran1, msec_ran2 = np.zeros(lstims), np.zeros(lstims)

for sindx, stim in enumerate(stims):
    val = np.where((dat_ran[sindx][base] > 4.5)*(dat_gcl[sindx][base] > 4.5))
#    print(val[0].shape)
    set1, set2 = np.where(dat_gcl[sindx][Tprim][val]>10), np.where(dat_gcl[sindx][Tprim][val]<10)
#    print(len(set1[0]), len(set2[0]))
    slp_con[sindx] = reg(dat_con[sindx][base], dat_con[sindx][Tprim])
    slp_gcl[sindx] = reg(dat_gcl[sindx][base][val], dat_gcl[sindx][Tprim][val])
    #######################
    slp_ran1[sindx] = reg(dat_ran[sindx][base][val][set1], dat_ran[sindx][Tprim][val][set1])
    slp_ran2[sindx] = reg(dat_ran[sindx][base][val][set2], dat_ran[sindx][Tprim][val][set2])
    #######################
    msec_ran2[sindx] = np.mean(dat_ran[sindx][secpeak][val][set2])
    msec_ran1[sindx] = np.mean(dat_ran[sindx][secpeak][val][set1])
    #######################
#######################
########################################
plt.close('all')
fig = plt.figure(figsize=[12, 5.5])
gm = gridspec.GridSpec(100, 160)

ax1 = plt.subplot(gm[:43, :45])
ax2 = plt.subplot(gm[:43, 55:100])
ax3 = plt.subplot(gm[:43, 115:160])

ax12 = plt.subplot(gm[57:, :45])
ax22 = plt.subplot(gm[57:, 55:100])
ax32 = plt.subplot(gm[57:, 115:160])

ax12.plot(0.001*np.array(stims), slp_con, '.k')
ax12.plot(0.001*np.array(stims), slp_gcl, '.', color = 'orange')

ax22.plot(0.001*np.array(stims), slp_ran1, '.b')
ax22.plot(0.001*np.array(stims), slp_ran2, '.r')

ax32.plot(0.001*np.array(stims), msec_ran1, '.b')
ax32.plot(0.001*np.array(stims), msec_ran2, '.r')

##############################
st = 3
val = np.where((dat_ran[st][base] > 4.5)*(dat_gcl[st][base] > 4.5))
set1, set2 = np.where(dat_gcl[st][Tprim][val]>10), np.where(dat_gcl[st][Tprim][val]<10)
plot_reg(ax1, dat_con[st][base], dat_con[st][Tprim], '0.3', 0.05)
plot_reg(ax1, dat_gcl[st][base][val], dat_gcl[st][Tprim][val], 'orange', 0.45)
#######################
plot_reg(ax2, dat_ran[st][base][val][set1], dat_ran[st][Tprim][val][set1], 'b', 0.05)
plot_reg(ax2, dat_ran[st][base][val][set2], dat_ran[st][Tprim][val][set2], 'r', 0.45)
#######################
sec_resp(ax3, dat_ran[st][secpeak][val][set2], dat_ran[st][tsecpeak][val][set2], 'r')
sec_resp(ax3, dat_ran[st][secpeak][val][set1], dat_ran[st][tsecpeak][val][set1], 'b')

for ax in [ax1, ax2, ax3, ax12, ax22, ax32]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    

for ax in [ax1, ax2]:
    ax.set_xlabel('ISI (ms)')
    ax.set_ylim(0, 39)
    ax.set_xlim(10, 200)
ax3.set_xlim(0, 20)
ax3.set_ylim(0, 120)

ax1.set_ylabel(r'$1^{o}$' + ' Resp Width (ms)')
ax3.set_xlabel(r'$2^{o}$' + ' Response Peak Ampl (spk/s)')
ax3.set_ylabel(r'$2^{o}$' + ' Response TTP (ms)')


ax12.set_ylabel(r'$1^{o}$' + ' Resp Fractional Width')
ax32.set_xlabel('Gstim (nS)')
ax32.set_ylabel(r'$2^{o}$' + ' Response Peak Ampl (spk/s)')


for ax in [ax12, ax22]:
    ax.set_xlabel('Gstim (nS)')
    ax.set_ylim(0, 0.2)
#    a.set_xlim(10, 200)
#for ax in axes[0]: ax.set_ylabel(r'$1^{o}$' + ' Resp Fractional Width')

plt.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.99)
plt.savefig('./summary_%s.png'%network, dpi=300)
