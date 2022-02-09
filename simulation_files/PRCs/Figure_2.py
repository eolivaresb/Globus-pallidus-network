import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from scipy import stats
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
################################################################
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

######################### 3 components fucntion to fit PRCs #######################################
def func(x, c1, c2, c3, e1, e2, e3, p_thre):
    comp1 = (x**e1*(p_thre**e1 - x**e1))/(e1*p_thre**(1+2*e1)/(1 + 3*e1+2*e1**2))
    comp2 = (x**e2*(p_thre**e2 - x**e2))/(e2*p_thre**(1+2*e2)/(1 + 3*e2+2*e2**2))
    comp3 = (x**e3*(p_thre**e3 - x**e3))/(e3*p_thre**(1+2*e3)/(1 + 3*e3+2*e3**2))
    y = c1*comp1 + c2*comp2 + c3*comp3
    y[np.where(y<0)] = 0.0
    return y
def comp(n, x, c1, c2, c3, e1, e2, e3, p_thre):
    comp1 = (x**e1*(p_thre**e1 - x**e1))/(e1*p_thre**(1+2*e1)/(1 + 3*e1+2*e1**2))
    comp2 = (x**e2*(p_thre**e2 - x**e2))/(e2*p_thre**(1+2*e2)/(1 + 3*e2+2*e2**2))
    comp3 = (x**e3*(p_thre**e3 - x**e3))/(e3*p_thre**(1+2*e3)/(1 + 3*e3+2*e3**2))
    z = [c1*comp1, c2*comp2, c3*comp3]
    y = z[n]
    y[np.where(y<0)] = 0.0
    return y

################################################################
###############################
## PRCs data
prcs = np.loadtxt('./prcs_data.txt')
phases = prcs[-1]
params = np.loadtxt('params_prc.txt') ## file with parameters from fitted prc done in PRC_fitting.py
rates = np.loadtxt('mean_rates.txt')
meanprcs = np.sum(prcs[:-1], axis = 1)/len(prcs.T)
prc_mass_center = np.array([np.sum(prcs[i]*prcs[-1])/np.sum(prcs[i]) for i in range(19)])
prc_shape = [meanprcs, prc_mass_center]
###################################################################
def in_hull(p, hull):
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0
##################################################################

plt.close('all')
fig = plt.figure(figsize = [9,5.5])
gm = gridspec.GridSpec(175, 85)

axr = [plt.subplot(gm[:32, i*30:i*30+23]) for i in range(3)]
axprc = [plt.subplot(gm[50:82, i*30:i*30+23]) for i in range(3)]
ax1 = plt.subplot(gm[95:,:35], projection ='3d')
ax2 = plt.subplot(gm[95:, 40:75], projection ='3d')

####################
### PLot PRC example with area and center of mass
####################
nex = 13
axr[0].plot(phases, prcs[nex], 'dk', mfc = 'w', ms = 3)
axr[0].fill_between(phases, prcs[nex], color = 'y', alpha = 0.24)

arrowprops=dict(arrowstyle='simple, tail_width=0.08,head_width=0.34,head_length=0.34', facecolor = 'k')
axr[0].annotate('$R(Z(\phi)) = %.2f$'%prc_mass_center[nex], xy=(prc_mass_center[nex], 1.25), ha = 'center', xytext=(prc_mass_center[nex], 2.2), arrowprops=arrowprops)
eq = r'$\int ~Z(\phi) = %.2f$'%(meanprcs[nex])
axr[0].text(0.5, 0.28, eq, {'color': 'k', 'fontsize': 11})
axr[0].set_ylim(0, 2.4)
axr[0].set_xlim(-0.01, 1.01)
axr[0].set_xlabel('Phase')
axr[0].set_ylabel('Cycles (pA s)'+'$^{-1}$')
####################
### PLot rate v/s PRCs shapes
####################
ylabels = [r'$\int ~Z(\phi)$', '$R(Z(\phi))$']
for i in range(2):
    x, y = rates[np.argsort(rates)], prc_shape[i][np.argsort(rates)]
    yp = np.poly1d(np.polyfit(x, y, 1))(x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    axr[i+1].plot(x, y, '.k')
    axr[i+1].plot(x, yp, 'k')
    axr[i+1].text(0.7, 0.92, 'p = %.2f'%p_value, transform=axr[i+1].transAxes)
    axr[i+1].set_xlabel('Firing rate [Hz]')
    axr[i+1].set_ylabel(ylabels[i])
    axr[i+1].set_xlim(0, 55)
axr[1].set_ylim(0, 2)
axr[2].set_ylim(0, 1)

for ax in [axr[0]] + axprc:
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1'])

fz = 12
####################
### PLot Prcs
####################
prcs_to_plot = [8, 7, 1]
xphase = np.linspace(0, 1, 1001)
for i, n in enumerate(prcs_to_plot):
    for j in range(3): axprc[i].plot(xphase, comp(j, xphase, *params[:,n]))
    axprc[i].plot(xphase, func(xphase, *params[:,n]), 'k', lw = 1.22)
    axprc[i].plot(phases, prcs[n], '-dk', mfc = 'w', lw = 0.3, ms = 3)
    axprc[i].set_xlim(-0.01, 1.01)
    axprc[i].set_ylim(-0.01, 1.82)
    axprc[i].set_xlabel('Phase')
    axprc[i].set_ylabel('Cycles (pA s)'+'$^{-1}$')

##################################################################
areas = np.array([np.array([params[0][j], params[1][j], params[2][j]]) for j in np.arange(len(params[0]))])
exponents = np.array([np.array([params[3][j], params[4][j], params[5][j]]) for j in np.arange(len(params[0]))])

####################
### PLot convex hull
####################
ax1.plot(areas.T[0], areas.T[1], areas.T[2], 'ro')
ax2.plot(exponents.T[0], exponents.T[1], exponents.T[2], 'ro')

ax1.set_xlabel('C1'+u'\u2801', fontsize = fz)
ax1.set_ylabel('C2', fontsize = fz)
ax1.set_zlabel('C3', fontsize = fz)
ax2.set_xlabel('e1', fontsize = fz)
ax2.set_ylabel('e2', fontsize = fz)
ax2.set_zlabel('e3', fontsize = fz)


areashull = ConvexHull(areas)
vertices = [areas[s] for s in areashull.simplices]
triangles = Poly3DCollection(vertices, edgecolor='k', alpha = 0.12, lw = 0.3)
ax1.add_collection3d(triangles)
ax1.set_xticks([0.3, 0.4, 0.5, 0.6])
ax1.set_yticks([0.1, 0.2, 0.3])
ax1.set_zticks([0.1, 0.2, 0.3])

exponentshull = ConvexHull(exponents)
vertices = [exponents[s] for s in exponentshull.simplices]
triangles = Poly3DCollection(vertices, edgecolor='k', alpha = 0.12, lw = 0.3)
ax2.add_collection3d(triangles)
ax2.set_xticks([0.4, 0.7, 1.0])
ax2.set_yticks([2, 4, 6])
ax2.set_zticks([20, 30, 40, 50])

for ax in axprc+axr:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

################################################################
x1, x2, x3, x4, x5, y0, y1, y2, fz = 0.02, 0.33, 0.67, 0.09, 0.53, 0.98, 0.73, 0.40, 16
plt.figtext(x1, y0, 'A', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x2, y0, 'B', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x3, y0, 'C', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x1, y1, 'D', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x2, y1, 'E', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x3, y1, 'F', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x4, y2, 'G', ha = 'center', va = 'center', fontsize = fz)
plt.figtext(x5, y2, 'H', ha = 'center', va = 'center', fontsize = fz)
################################################################
fig.subplots_adjust(left = 0.06, bottom = 0.03, right = 0.99, top = 0.98)
plt.savefig('figure2.png', dpi = 400)
# plt.savefig('figure2.pdf')
