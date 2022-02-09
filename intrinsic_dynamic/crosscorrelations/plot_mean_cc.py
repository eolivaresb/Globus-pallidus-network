import numpy as np
import time as clocktime
################################################################
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
################################################################
N = 1000
################################################################
import matplotlib
matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
###############################################################

anatomy_categories = ['Self', u'\u24D8\u2194\u24D9',u'\u24D8\u2192\u24D9', u'\u24D8\u2192\u24de\u2192\u24D9', u'\u24D8\u2190\u24de\u2192\u24D9', 'None']
################################################################
plt.close('all')
fig = plt.figure(figsize = [6,8])
gm = gridspec.GridSpec(185, 200, figure = fig)
axes = [[plt.subplot(gm[j*40:j*40+25, i*100:i*100+80]) for j in range(5)] for i in range(2)]
################################################################
ttot = 1800
twindow, tbin = 0.5, 0.001
timecc = np.arange(-twindow, twindow+tbin, tbin)

for col, net in enumerate(['small', 'random']):
    ###############################################################
    ccorr = np.load('cc_%s_10.npy'%net, allow_pickle = 'true')
    z =  np.loadtxt('../../simulation_files/network_matrix_%s.dat'%net).astype(int)
    rates = np.loadtxt('../%s_10/rates_and_cv.dat'%(net))[0]
    ##############################################################
    mean_cc = [np.zeros(1001) for i in range(len(anatomy_categories[1:]))]
    pair_indx = -1
    npairs = np.zeros(len(anatomy_categories[1:]))
    ###########################################################
    for i in range(N):
        for j in range(i, N):
            pair_indx +=1
            if ((rates[i]>2)*(rates[j]>2)*(i!=j)):
                if ((z[i,j]==1)*(z[j,i]==1)):  # i to j and j to i
                    mean_cc[0] += ccorr[pair_indx]
                    npairs[0]  +=1
                elif ((z[i,j]==1)): # at least one connection between i and j
                    mean_cc[1] += ccorr[pair_indx]
                    npairs[1]  +=1
                elif ((z[j,i]==1)): # at least one connection between i and j
                    mean_cc[1] += ccorr[pair_indx][::-1]
                    npairs[1]  +=1
                elif(np.sum(z[i]*z[:,j])*np.sum(z[:,i]*z[j])): # one neuron connect i to j, same or another connect j to i, then bigges firing rate determine ccorr directionality
                    if (rates[i]>rates[j]):
                        mean_cc[2] += ccorr[pair_indx]
                    else:
                        mean_cc[2] += ccorr[pair_indx][::-1]
                    npairs[2]  +=1
                elif(np.sum(z[i]*z[:,j])): # one neuron connect i to j
                    mean_cc[2] += ccorr[pair_indx]
                    npairs[2]  +=1
                elif(np.sum(z[:,i]*z[j])): # one neuron connect j to i
                    mean_cc[2] += ccorr[pair_indx][::-1]
                    npairs[2]  +=1
                elif(np.sum(z[:,i]*z[:,j])): #at least one neuron project to both i, j
                    mean_cc[3] += ccorr[pair_indx]
                    npairs[3]  +=1
                else:                               # none of the above categories
                    mean_cc[4] += ccorr[pair_indx]
                    npairs[4]  +=1
    np.save('mean_cc_%s.npy'%net, [mean_cc, npairs])
    [mean_cc, npairs] = np.load('mean_cc_%s.npy'%net, allow_pickle = 'true')

    ###################################################
    for k, categ in enumerate(anatomy_categories[1:]):
        axes[col][k].plot(1000*timecc, mean_cc[k]/npairs[k], color = 'k', label = anatomy_categories[k+1])
        axes[col][k].set_title('%s: number of pairs = %d'%(anatomy_categories[k+1], npairs[k]))
###################################################
for ax in axes[0]+axes[1]:
    ax.axvline(x = 0, linestyle = '--', linewidth = 0.2, color = 'k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_xlim(-100, 100)
#    ax.set_ylim(0, 26)
for ax in [axes[0][4], axes[1][4]]:
    ax.set_xticklabels(['%d'%d for d in [-100, -50, 0, 50, 100]])
    ax.set_xlabel('Time lag (ms)')

plt.figtext(0.25, 0.97, 'Small-world network', fontsize = 16, ha = 'center')
plt.figtext(0.75, 0.97, 'Random network', fontsize = 16, ha = 'center')

fig.subplots_adjust(left = 0.1, bottom = 0.06, right = 0.98, top = 0.92)
#plt.savefig('Mean_cc.png', dpi = 300)
plt.savefig('Mean_cc_noscaled.png', dpi = 300)
