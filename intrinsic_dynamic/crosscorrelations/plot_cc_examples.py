import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
################################################################
N = 1000
################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
################################################################
anatomy_categories = ['Self', u'\u24D8\u2194\u24D9',u'\u24D8\u2192\u24D9', u'\u24D8\u2192\u24de\u2192\u24D9', u'\u24D8\u2190\u24de\u2192\u24D9', 'None']
################################################################
ttot = 1800
twindow, tbin = 0.5*1000, 0.001*1000
timecc = np.arange(-twindow, twindow+tbin, tbin)

 ##############################################################
def plot_cc(i, j, ax, direction):
    position = i/2*(1999-i) +j
    cc = ccorr[int(position)]
    if direction=='f':ax.plot(timecc, cc, color = 'k')
    if direction=='r':ax.plot(timecc, cc[::-1], color = 'k')

###############################################################
for net in ['small', 'random']:
    ccorr = np.load('cc_%s.npy'%net, allow_pickle = 'true')

    z =  np.loadtxt('../../simulation_files/network_matrix_%s.dat'%net).astype(int)
    rates = np.loadtxt('../%s/rates_and_cv.dat'%(net))[0]
    isflat = np.loadtxt('data/Isflat_%s'%net)
    anatomy = np.loadtxt('data/anatomy_categories_%s.txt'%net)

    ################################################################
    ################################################################
    plt.close('all')
    fig = plt.figure(figsize = [6,7])
    gm = gridspec.GridSpec(185, 200, figure = fig)
    ax = plt.subplot(gm[:,:])
    axes = [plt.subplot(gm[j*40:j*40+25, i*100:i*100+80]) for j in range(5) for i in range(2)]
    tuples = np.where(anatomy==1)
    ntuples = len(tuples[0])
    n=0
    for k in range(100):
        indx = np.random.randint(ntuples) #random couple
        a,b = tuples[0][indx], tuples[1][indx]
        i,j = np.min([a,b]), np.max([a,b])
        if (isflat[i, j]==1):
            print(n, i, j, isflat[i,j])
            if rates[i] > rates[j]: plot_cc(i,j, axes[n], 'f')
            else: plot_cc(i,j, axes[n], 'r')
            n+=1
        if n==10:break
    for k, ax in enumerate(axes):
        ax.axvline(x = 0, linestyle = '--', linewidth = 0.2, color = 'k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_xlim(-100, 100)
        if k%2==0: ax.set_ylabel('Rate (Hz)', fontsize = 9)
        if k in [8, 9]:
            ax.set_xlabel('Time (ms)', fontsize = 9)
            ax.set_xticklabels([-100, -50, 0, 50, 100])
    plt.figtext(0.5, 0.96, net + '  %s'%anatomy_categories[1], fontsize = 24, ha = 'center')
    fig.subplots_adjust(left = 0.12, bottom = 0.06, right = 0.98, top = 0.9)
    plt.savefig('CC_examples_%s_mutual.png'%net, dpi = 300)
    ################################################################
    #################################################################
    plt.close('all')
    fig = plt.figure(figsize = [6,7])
    gm = gridspec.GridSpec(185, 200, figure = fig)
    ax = plt.subplot(gm[:,:])
    axes = [plt.subplot(gm[j*40:j*40+25, i*100:i*100+80]) for j in range(5) for i in range(2)]
    tuples = np.where(anatomy==2)
    ntuples = len(tuples[0])
    n=0
    for k in range(100):
        indx = np.random.randint(ntuples) #random couple
        a,b = tuples[0][indx], tuples[1][indx]
        i,j = np.min([a,b]), np.max([a,b])
        if (isflat[i, j]==1):
            print(n, i, j, isflat[i,j])
            if z[i,j]==1: plot_cc(i,j, axes[n], 'f')
            if z[j,i]==1: plot_cc(i,j, axes[n], 'r')
            n+=1
        if n==10:break
    for k, ax in enumerate(axes):
        ax.axvline(x = 0, linestyle = '--', linewidth = 0.2, color = 'k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_xlim(-100, 100)
        if k%2==0: ax.set_ylabel('Rate (Hz)', fontsize = 9)
        if k in [8, 9]:
            ax.set_xlabel('Time (ms)', fontsize = 9)
            ax.set_xticklabels([-100, -50, 0, 50, 100])
    plt.figtext(0.5, 0.96, net + '  %s'%anatomy_categories[2], fontsize = 24, ha = 'center')
    fig.subplots_adjust(left = 0.12, bottom = 0.06, right = 0.98, top = 0.9)
    plt.savefig('CC_examples_%s_unidirectional.png'%net, dpi = 300)
    #################################################################
    #################################################################
    plt.close('all')
    fig = plt.figure(figsize = [6,7])
    gm = gridspec.GridSpec(185, 200, figure = fig)
    ax = plt.subplot(gm[:,:])
    axes = [plt.subplot(gm[j*40:j*40+25, i*100:i*100+80]) for j in range(5) for i in range(2)]
    tuples = np.where(anatomy==3)
    ntuples = len(tuples[0])
    n=0
    for k in range(300):
        indx = np.random.randint(ntuples) #random couple
        a,b = tuples[0][indx], tuples[1][indx]
        i,j = np.min([a,b]), np.max([a,b])
        if (isflat[i, j]==1):
            print(n, i, j, isflat[i,j])
            if np.sum(z[i,:]*z[:,j])>=1:
                axes[n].text(0.1, 1.00, '#(i>o>j) = %d'%np.sum(z[i,:]*z[:,j]), transform=axes[n].transAxes)
            if np.sum(z[j,:]*z[:,i])>=1:
                axes[n].text(0.6, 1.00, '#(j>o>i) = %d'%np.sum(z[j,:]*z[:,i]), transform=axes[n].transAxes)
            if np.sum(z[:,j]*z[:,i])>=1:
                axes[n].text(0.35, 1.24, '#(j<o>i) = %d'%np.sum(z[:,j]*z[:,i]), transform=axes[n].transAxes)
            plot_cc(i,j, axes[n], 'f')
            n+=1
        if n==10:break
    for k, ax in enumerate(axes):
        ax.axvline(x = 0, linestyle = '--', linewidth = 0.2, color = 'k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_xlim(-100, 100)
        if k%2==0: ax.set_ylabel('Rate (Hz)', fontsize = 9)
        if k in [8, 9]:
            ax.set_xlabel('Time (ms)', fontsize = 9)
            ax.set_xticklabels([-100, -50, 0, 50, 100])
    plt.figtext(0.5, 0.96, net + '  %s'%anatomy_categories[3], fontsize = 24, ha = 'center')
    fig.subplots_adjust(left = 0.12, bottom = 0.06, right = 0.98, top = 0.9)
    plt.savefig('CC_examples_%s_two_synapses_non_flat.png'%net, dpi = 300)
    #################################################################
    #################################################################
    plt.close('all')
    fig = plt.figure(figsize = [6,7])
    gm = gridspec.GridSpec(185, 200, figure = fig)
    ax = plt.subplot(gm[:,:])
    axes = [plt.subplot(gm[j*40:j*40+25, i*100:i*100+80]) for j in range(5) for i in range(2)]
    tuples = np.where(anatomy==4)
    ntuples = len(tuples[0])
    n=0
    for k in range(3000):
        indx = np.random.randint(ntuples) #random couple
        a,b = tuples[0][indx], tuples[1][indx]
        i,j = np.min([a,b]), np.max([a,b])
        if (isflat[i, j]==1):
            print(n, i, j, isflat[i,j])
            if np.sum(z[:,j]*z[:,i])>=1:
                axes[n].text(0.35, 1.24, '#(j<o>i) = %d'%np.sum(z[:,j]*z[:,i]), transform=axes[n].transAxes)
            plot_cc(i,j, axes[n], 'f')
            n+=1
        if n==10:break
    for k, ax in enumerate(axes):
        ax.axvline(x = 0, linestyle = '--', linewidth = 0.2, color = 'k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticklabels([])
        ax.set_xlim(-100, 100)
        if k%2==0: ax.set_ylabel('Rate (Hz)', fontsize = 9)
        if k in [8, 9]:
            ax.set_xlabel('Time (ms)', fontsize = 9)
            ax.set_xticklabels([-100, -50, 0, 50, 100])
    plt.figtext(0.5, 0.96, net + '  %s'%anatomy_categories[4], fontsize = 24, ha = 'center')
    fig.subplots_adjust(left = 0.12, bottom = 0.06, right = 0.98, top = 0.9)
    plt.savefig('CC_examples_%s_shared_inputs_non_flat.png'%net, dpi = 300)
    #################################################################
    #################################################################
#plt.close('all')
#fig = plt.figure(figsize = [6,7])
#gm = gridspec.GridSpec(185, 200, figure = fig)
#ax = plt.subplot(gm[:,:])
#axes = [plt.subplot(gm[j*40:j*40+25, i*100:i*100+80]) for j in range(5) for i in range(2)]
#tuples = np.where(anatomy==3)
#ntuples = len(tuples[0])
#n=0
#for k in range(100):
#    indx = np.random.randint(ntuples) #random couple
#    a,b = tuples[0][indx], tuples[1][indx]
#    i,j = np.min([a,b]), np.max([a,b])
#    if (isflat[i, j]==0)*(rates[i]>2)*(rates[j]>2):
#        print(n, i, j, isflat[i,j])
#        plot_cc(i,j, axes[n])
#        n+=1
#    if n==10:break
#for k, ax in enumerate(axes):
#    ax.axvline(x = 0, linestyle = '--', linewidth = 0.2, color = 'k')
#    ax.spines['right'].set_visible(False)
#    ax.spines['top'].set_visible(False)
#    ax.set_xticklabels([])
#    ax.set_xlim(-100, 100)
#    if k%2==0: ax.set_ylabel('Rate (Hz)', fontsize = 9)
#    if k in [8, 9]: ax.set_xlabel('Time (ms)', fontsize = 9)
#plt.figtext(0.5, 0.96, '%s'%anatomy_categories[3], fontsize = 24, ha = 'center')
#fig.subplots_adjust(left = 0.12, bottom = 0.06, right = 0.98, top = 0.92)
#plt.savefig('CC_examples_two_synapses_flat.png', dpi = 300)
