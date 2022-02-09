import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
######################### 3 components fucntion to fit PRCs #######################################
def func(x, c1, c2, c3, e1, e2, e3, p_thre):
    comp1 = (x**e1*(p_thre**e1 - x**e1))/(e1*p_thre**(1+2*e1)/(1 + 3*e1+2*e1**2))
    comp2 = (x**e2*(p_thre**e2 - x**e2))/(e2*p_thre**(1+2*e2)/(1 + 3*e2+2*e2**2))
    comp3 = (x**e3*(p_thre**e3 - x**e3))/(e3*p_thre**(1+2*e3)/(1 + 3*e3+2*e3**2))
    y = c1*comp1 + c2*comp2 + c3*comp3
    y[np.where(y<0)] = 0.0
    return y
################################################################
###############################
###################################################################
def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

##################################################################
params = np.loadtxt('params_prc.txt') ## file with parameters from fitted prc done in PRC_fitting.py

areas = np.array([  np.array(  [params[0][j], params[1][j], params[2][j]  ]) for j in np.arange(len(params[0]))])
exponents = np.array([  np.array(  [params[3][j], params[4][j], params[5][j]  ]) for j in np.arange(len(params[0]))])

minimas = np.min(params, axis = 1)
maximas = np.max(params, axis = 1)

###################################################################
############# Generate random prc from params uniform distibution
############# Add them to generated prcs files only if they fit
############# in to the convex hull defined by: component areas; component exponents
###################################################################
randomprc = []
sel = 0
for k in range(50000): # some random point will not fall in the hull, 50000 trials are enough to have 1000 points on it
    rarea = np.array([np.random.uniform(minimas[i], maximas[i]) for i in[0, 1, 2]])
    rexp = np.array([np.random.uniform(minimas[i], maximas[i]) for i in[3, 4, 5]])
    rpthr = np.random.uniform(minimas[6], maximas[6])
    if in_hull(rarea, areas):
        if in_hull(rexp, exponents):
            sel+=1
            print(sel, k, sel/k*100)
            randomprc.append([rarea[0], rarea[1], rarea[2], rexp[0], rexp[1], rexp[2], rpthr])
            if (sel == 1000): break
np.savetxt('diff_prc.dat', np.array(randomprc))

###################################################################
################################################################

##### plot a subset of generated prcs
# import matplotlib.pylab as plt
# import matplotlib.gridspec as gridspec

#randomprc = np.loadtxt('diff_prc.dat')
#phase = np.linspace(0, 1, 10000)

#for j in range(10):
#    plt.close('all')
#    fig = plt.figure(figsize = [8,12])
#    gm = gridspec.GridSpec(100, 100)
#    axes = [plt.subplot(gm[i*20:i*20+15, k*50:k*50+45]) for i in range(5) for k in range(2)]
#    for i in range(10):
#        popt = randomprc[10*j+i]
#        axes[i].plot(phase, func(phase, *popt), 'b')
#        for prcfit in params.T:
#            axes[i].plot(phase, func(phase, *prcfit), 'r', lw = 0.2)
#
#
#    fig.subplots_adjust(left = 0.08, bottom = 0.03, right = 0.95, top = 0.98)
#    plt.savefig('generated/generated_prc_%d.png'%j)
