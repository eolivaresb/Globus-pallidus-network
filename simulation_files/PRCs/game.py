import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from matplotlib.widgets import Button
import time as clocktime
################################################################
axcolor = 'lightgoldenrodyellow'
'''
A game to visually inspect generated PRCs:
Three prcs are presented, one of them is generate and 2 comes from the experimental sample.
In most cases, the generated looks different but is not possible to be idenetified as artificial
'''

###############################################################
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
## PRCs data
params = np.loadtxt('params_prc.txt').T ## file with parameters from fitted prc done in PRC_fitting.py
xphase = np.linspace(0, 1, 101)
artificial = np.loadtxt('diff_prc.dat')
###################################################################
##################################################################
plt.close('all')
fig = plt.figure(figsize = [12,4.5])
gm = gridspec.GridSpec(35, 85)
axr = [plt.subplot(gm[8:30, i*30:i*30+23]) for i in range(3)]
text1 = axr[0].text(0.5, 5.3, 'A\n', fontsize = 22, va = 'bottom', ha = 'center')
text2 = axr[1].text(0.5, 5.3, 'B\n', fontsize = 22, va = 'bottom', ha = 'center')
text3 = axr[2].text(0.5, 5.3, 'C\n', fontsize = 22, va = 'bottom', ha = 'center')
letters = ['A', 'B', 'C']
typen = ['real', 'real', 'artf']

class Index(object):
    def reset_reveal(self, event):
        texts = [text1, text2, text3]
        axr[rda[2]].fill_between(xphase, func(xphase, *artificial[na]), color = 'r', alpha = 0.5)
        texts[rda[0]].set_text(letters[rda[0]]+'\n'+'PRC real %d'%(1+nr1))
        texts[rda[1]].set_text(letters[rda[1]]+'\n'+'PRC real %d'%(1+nr2))
        texts[rda[2]].set_text(letters[rda[2]]+'\n'+'PRC artf %d'%(1+na))

    def reset(self, event):
        global rda, nr1, nr2, na, text1, text2, text3
        for i, ax in enumerate(axr):
            ax.clear()
            ax.set_ylim(0, 5)
            ax.set_xlim(0, 1)
        [nr1, nr2] = np.random.choice(np.arange(19), 2, replace = False)
        na = np.random.choice(np.arange(1000), 1)[0]
        prcp = [params[nr1], params[nr2], artificial[na]]
        rda = np.random.choice(np.arange(3), 3, replace = False)
        axr[rda[0]].plot(xphase, func(xphase, *prcp[0]), 'k', lw = 2)
        axr[rda[1]].plot(xphase, func(xphase, *prcp[1]), 'k', lw = 2)
        axr[rda[2]].plot(xphase, func(xphase, *prcp[2]), 'k', lw = 2)
        text1 = axr[0].text(0.5, 5.3, 'A\n', fontsize = 22, va = 'bottom', ha = 'center')
        text2 = axr[1].text(0.5, 5.3, 'B\n', fontsize = 22, va = 'bottom', ha = 'center')
        text3 = axr[2].text(0.5, 5.3, 'C\n', fontsize = 22, va = 'bottom', ha = 'center')

callback = Index()

revealprc = plt.axes([0.5, 0.025, 0.1, 0.04])
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button_reveal = Button(revealprc, 'Reveal')
button_reset = Button(resetax, 'Reset')

button_reveal.on_clicked(callback.reset_reveal)
button_reset.on_clicked(callback.reset)

fig.subplots_adjust(left = 0.06, bottom = 0.03, right = 0.99, top = 0.98)
plt.show()
