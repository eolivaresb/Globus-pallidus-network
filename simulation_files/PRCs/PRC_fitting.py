import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
import os
import glob
##################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##################################################################
def func(x, c1, c2, c3, e1, e2, e3, p_thre):
    comp1 = (x**e1*(p_thre**e1 - x**e1))/(e1*p_thre**(1+2*e1)/(1 + 3*e1+2*e1**2))
    comp2 = (x**e2*(p_thre**e2 - x**e2))/(e2*p_thre**(1+2*e2)/(1 + 3*e2+2*e2**2))
    comp3 = (x**e3*(p_thre**e3 - x**e3))/(e3*p_thre**(1+2*e3)/(1 + 3*e3+2*e3**2))
    y = c1*comp1 + c2*comp2 + c3*comp3
    y[np.where(y<0)] = 0.0
    return y
##################################################################
'''Read data files and formatted to be read in python'''
os.chdir('Matt_data')
files = glob.glob('*PRC.txt')
fitting = 0
os.system('perl -i -p -e \'s/}//g;\' *.txt')
os.system('perl -i -p -e \'s/{//g;\' *.txt')
os.system('perl -i -p -e \'s/\^//g;\' *.txt')
os.system('perl -i -p -e \'s/\*/e/g;\' *.txt')
os.system('perl -i -p -e \'s/,/\n/g;\' *.txt')
##################################################################
##################################################################
##############   Sort by firing rate      ########################
rates = np.zeros(19)
phase = np.linspace(0, 1, 10000)
for k, label in enumerate(files):
    exp_label = label[:-8]
    f = np.loadtxt('%s_PRC.txt'%exp_label)
    rates[k] = 1./f[-1]
frate_order = np.argsort(rates)
np.savetxt('mean_rates.txt', rates)
print('mean rate = %.2f; std  = %.2f'%(np.mean(rates), np.std(rates)))
    ##################################################################
    ##################################################################
    ##################################################################
if fitting:
    x = phase
    prcs = []
    params_file = np.zeros((len(files), 7))
    #for k, label in enumerate([files[0]]):
    for k in range(19):
        label = files[frate_order[k]]
        exp_label = label[:-8]
        f = np.loadtxt('%s_PRC.txt'%exp_label)
        meanisi = f[-1]
        prc = np.vstack([f[:40], f[40:80]])

        prc_error = f[80:120]*1000
        prc[np.where(prc<0)]=0.0

        plt.close('all')
        fig = plt.figure(figsize = [8,4])
        gm = gridspec.GridSpec(20, 24)
        ax = plt.subplot(gm[1:18, 0:22])
        prc = prc[:,np.argsort(prc[0])]
        prc[1] = prc[1]*1000 # move from cycles/(mA*ms) to cycles/(mA*s)

        prcs.append(prc[1])
        ax.errorbar(prc[0], prc[1], yerr=prc_error, fmt = '-', color = 'red', zorder = 10)
        ax.plot(prc[0], prc[1], '.', color = 'red',zorder = 10)
        ax.axhline(y=0, linestyle = '--', linewidth = 0.3)

        #Fit for the parameters a, b, c of the function func:
        popt, pcov = curve_fit(func, prc[0], prc[1], p0 = [1, 1, 1, 1, 10, 30, 1], bounds=(0.0001, [50., 50., 50., 50., 50., 50., 1.0]) )
        params_file[k] = popt

        ax.plot(prc[0], func(prc[0], *popt), 'g-', label='c1=%5.1f, c2=%5.1f, c3=%5.1f, e1=%5.1f, e2=%5.1f, e3=%5.1f, p_thre=%5.3f' % tuple(popt), zorder = 1)
        ax.plot(phase, func(phase, *popt), '--b', lw = 1.6, zorder = 140)
        ax.set_xlim(-0.01, 1.01)
        ax.text(0.05, 0.88,'\nMean rate = %.1f (Hz)'%(1/meanisi), ha='left', va='center', transform = ax.transAxes)
        fig.subplots_adjust(left = 0.1, bottom = 0.05, right = 0.95, top = 0.90)
        [c1, c2, c3, e1, e2, e3, p_thre] = popt
        ax.set_xlabel('Phase')
        ax.set_ylabel('PRC  cycles/(mA*s)')
        ax.text(0.05, 0.93, 'Cell: %s'%exp_label, ha = 'left', va = 'center', transform = ax.transAxes, fontsize = 13)
        ax.plot(x, c1*(x**e1*(p_thre**e1 - x**e1))/(e1*p_thre**(1+2*e1)/(1 + 3*e1+2*e1**2)),'y', zorder = 0)
        ax.plot(x, c2*(x**e2*(p_thre**e2 - x**e2))/(e2*p_thre**(1+2*e2)/(1 + 3*e2+2*e2**2)), 'y', zorder = 0)
        ax.plot(x, c3*(x**e3*(p_thre**e3 - x**e3))/(e3*p_thre**(1+2*e3)/(1 + 3*e3+2*e3**2)), 'y', zorder = 0)
        ax.set_ylim(bottom = -0.2)

        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)

        plt.savefig('../prc_figures/%d_fitting_prc_%s_0.png'%(k+1, exp_label), dpi = 250)
    #os.chdir('../')
    prcs.append(prc[0])
    np.savetxt('../prcs_data.txt', np.array(prcs))
    np.savetxt('../params_prc.txt', params_file.T)



params = np.loadtxt('../params_prc.txt').T
prcs = []

plt.close('all')
fig = plt.figure(figsize = [15,8])
gm = gridspec.GridSpec(114, 144)
axes = [plt.subplot(gm[i*30:i*30+24, j*30:j*30+24]) for i in range(4) for j in range(5)]

R2_set = np.zeros(19)

for k in range(19):
    label = files[frate_order[k]]
    exp_label = label[:-8]
    f = np.loadtxt('%s_PRC.txt'%exp_label)
    meanisi = f[-1]
    prc = np.vstack([f[:40], f[40:80]])
    prc_error = f[80:120]*1000
    prc[np.where(prc<0)]=0.0
    prc = prc[:,np.argsort(prc[0])]
    prc[1] = prc[1]*1000 # move from cycles/(mA*ms) to cycles/(mA*s)
    
    popt = params[k]
    
    prc_data = prc[1]
    prc_fit = func(prc[0], *popt)
    
    r2 = 1 - np.sum((prc_data - prc_fit)**2) / np.sum((prc_data - np.mean(prc_data))**2)
    R2_set[k] = r2
    
    axes[k].plot(prc[0], prc[1], 'or', ms = 1.5)
    axes[k].errorbar(prc[0], prc[1], yerr=prc_error, color = 'r', lw = 0.95)#, fmt = '-', color = 'red')
    axes[k].axhline(y=0, linestyle = '--', linewidth = 0.3)


    axes[k].plot(prc[0], func(prc[0], *popt), 'ob', ms = 1.5, lw = 0.5)
    axes[k].plot(phase, func(phase, *popt), 'b', lw = 0.95, zorder = 12)
    axes[k].set_xlim(-0.01, 1.01)

    if k >13: axes[k].set_xlabel('Phase')
    if k in [0, 5, 10, 15]:
        axes[k].set_ylabel('PRC %d\nCycles/(mA*s)'%(k+1))
    else:
        axes[k].set_ylabel('PRC %d'%(k+1))
    axes[k].set_yticks([0, 1, 2, 3])
    axes[k].set_ylim(bottom = -0.2)
    axes[k].set_xticks([0, 0.5, 1])
    axes[k].set_xticklabels(['0', '0.5', '1'])

    axes[k].text(0.05, 0.75,'Mean rate = %.1f (Hz)\n R2 = %.2f'%(1/meanisi, r2), ha='left', va='center', transform = axes[k].transAxes)

    prcs.append([prc[0], prc[1], func(prc[0], *popt)])

for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
axes[-1].axis('off')

fig.subplots_adjust(left = 0.05, bottom = 0.08, right = 0.98, top = 0.97)
plt.savefig('../fitted_prc.png', dpi = 350)
os.chdir('../')
print ('Max r2 = %.2f, Min r2 =  = %.2f, Mean R2 = %.2f, std R2 = %.2f'%(np.max(R2_set), np.min(R2_set), np.mean(R2_set), np.std(R2_set)))
