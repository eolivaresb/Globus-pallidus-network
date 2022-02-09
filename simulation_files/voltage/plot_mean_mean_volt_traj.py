import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

v = np.loadtxt('GP_PV_Trajectories.txt')

a = []
a.append(v[0][range(0, 2002, 2)])

for i in range(19):
    a.append(v[i][range(1, 2002, 2)])
    
np.savetxt('voltage_traces.txt', np.array(a).T, fmt = '%.5f')
plt.close('all')

fig = plt.figure(figsize = [6,5])
v = np.loadtxt('voltage_traces.txt').T

vmean = np.zeros((1001, 2))
vmean[:,0] = v[0]
vmean[:,1] = np.mean(v[1:], axis = 0)
np.savetxt('volt.dat', vmean, fmt = '%.4f')
v[1:]*=1000

gm = gridspec.GridSpec(20, 500)
ax0 = plt.subplot(gm[1:-1, 1:])

for i in range(19):
    ax0.plot(v[0], v[1+i], 'g', lw = 0.5, alpha = 0.5)

meanv = np.mean(v[1:], axis = 0)

ax0.plot(v[0], meanv, 'k', lw = 2)

ax0.axhline(y = np.max(meanv), linestyle = '--', color = 'k')
ax0.text(0.3, 1+np.max(meanv), 'max of the mean voltage')
ax0.axhline(y = np.mean(np.max(v[1:], axis = 1)), linestyle = '-.', color = 'b') 
ax0.text(0.69, 1+np.mean(np.max(v[1:], axis = 1)), 'mean of the max voltage')

ax0.axhline(y = np.min(meanv), linestyle = '--', color = 'k')
ax0.text(0.3, 1+np.min(meanv), 'min of the mean voltage')
ax0.axhline(y = np.mean(np.min(v[1:], axis = 1)), linestyle = '-.', color = 'b') 
ax0.text(0.69, 1+np.mean(np.min(v[1:], axis = 1)), 'mean of the min voltage')

ax0.set_xlim(0, 1)
ax0.set_xlabel('Phase')
ax0.set_ylabel('Volt (mV)')
fig.subplots_adjust(left = 0.1, bottom = 0.05, right = 0.96, top = 0.98)
plt.savefig('GPe_PV_volt.png', dpi = 300)
#ax0.set_ylim(-0.07,-0.05)
######################
plt.close('all')
fig = plt.figure(figsize = [6,4])
gm = gridspec.GridSpec(20, 500)
ax0 = plt.subplot(gm[1:-1, 1:])

meanv = np.mean(v[1:], axis = 0)

ax0.plot(v[0], meanv, 'k', lw = 2)
ax0.set_ylim(-70,-50)
ax0.set_xlim(0, 1)
ax0.set_xlabel('Phase')
ax0.set_ylabel('Volt (mV)')
ax0.set_yticks([-70, -65, -60, -55, -50])

fig.subplots_adjust(left = 0.1, bottom = 0.05, right = 0.96, top = 0.98)
plt.savefig('GPe_PV_truncated_volt.png', dpi = 300)
