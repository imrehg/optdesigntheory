###
# Optimal design with Lorentzian lineshape
###

from __future__ import division
from numpy import *
import pylab as pl
from scipy.odr import *
from optimaldesign import *

dlx0 = lambda p, x: 2*(x-p[0])*p[2]*p[1]/((x-p[0])**2+p[1]**2)**2
dlg = lambda p, x: p[2]*((x-p[0])**2 - p[1]**2)/((x-p[0])**2+p[1]**2)**2
dla = lambda p, x: p[1]/((x-p[0])**2+p[1]**2)
dlc = lambda p, x: x*0+1

def getf(p, x):
    return array([[dlx0(p, x)],[dlg(p,x)],[dla(p,x)]])
    # return array([[dlg(p, x)],[dla(p,x)],[dlx0(p,x)]])
    # return array([[dlx0(p, x)],[dlg(p,x)],[dla(p,x)], [dlc(p, x)]])
    # return array([[dlx0(p, x)],[dlg(p,x)],[dla(p,x)]])
    # return array([[dlg(p, x)],[dlx0(p,x)],[dla(p,x)]])
    # return array([[dla(p, x)],[dlx0(p,x)],[dlg(p,x)]])

lorentz = lambda p, x: p[2] * p[1] / ((x-p[0])**2 + p[1]**2)

nfig = 0
############# Figure: d(x, xi) vs x
nfig += 1
p = [0, 1, 1]
xi = array([[-0.775, 0, 0.775], [1/3, 1/3, 1/3]])
M1 = Mcontinuous(getf, p, xi)
xlim = [-1.5, 1.5]

x = linspace(xlim[0], xlim[1],201)
dx = sdvarcont(getf, p, xi, x, optimize=False)

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pl.rcParams.update(params)
pl.figure(1)
pl.clf()
ymarg = 0.160
xmarg = 0.2
pl.axes([ymarg,xmarg,0.95-ymarg,0.95-xmarg])
pl.plot(x,dx,'k-')
pl.plot(xlim,[3,3],'b--')
# pl.plot(xi[0,:],[3]*3, 'r.', markersize=12, label='Design points')
pl.ylim([min(dx), 3.2])
pl.xlabel(r'$x$')
pl.ylabel(r'$d(x, \xi^*)$')
# pl.legend(loc='lower center')
pl.savefig('fig1.eps')

############# Figure: D_eff vs N
p = [0, 1, 1]
xi = array([[-0.775, 0, 0.775], [1/3, 1/3, 1/3]])
M1 = Mcontinuous(getf, p, xi)
xlim = [-1.5, 1.5]

# x = linspace(xlim[0], xlim[1],201)
# dx = sdvarcont(getf, p, xi, x, optimize=False)

xin = array([1, 0, 1])
mmax = 30
xout = sequential(getf, p, xin, xlim, mmax, optimize=False)

tt = range(4, mmax)
res = array([])
for step in tt:
    xx = xout[:step]
    M = Mexact(getf, p, xx)
    res = append(res, (det(M)/det(M1))**(1/3))

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pl.rcParams.update(params)
pl.figure(1)
pl.clf()
ymarg = 0.160
xmarg = 0.2
pl.axes([ymarg,xmarg,0.95-ymarg,0.95-xmarg])
pl.plot(tt, res, 'k.')
pl.xlabel(r'$N$')
pl.ylabel(r'$D_\mathrm{eff}$')
pl.savefig('fig2.eps')


############# Figure: D_eff vs dx0

def uniform(x, pars):
    prob = 1 / abs(pars[1] - pars[0])
    return prob

p = [0, 1, 1]
xi = array([[-0.775, 0, 0.775], [1/3, 1/3, 1/3]])
xlim = [-3, 3]

xdif = linspace(0, 0.8, 31)
res = array([])
for xd in xdif:
    xi = array([[-0.775-xd, 0-xd, 0.775-xd], [1/3, 1/3, 1/3]])
    M1 = Mcontinuous(getf, p, xi)
    res = append(res, det(M1))

def compoundexp(x, pars):
    """
    e.g. pars = ((-a, b, 1), (a, b, 1))
    """
    h = 0
    for par in pars:
        try:
            prob += par[2]*exp(-(x-par[0])**2/(2*par[1]**2))/sqrt(2*pi*par[1])
        except NameError:
            prob = par[2]*exp(-(x-par[0])**2/(2*par[1]**2))/sqrt(2*pi*par[1])
        h += par[2]
    prob = prob /h
    return prob

res2 = array([])
for xd in xdif:
    a = 0.775
    b = 0.4
    pars = [[-a-xd, b, 1], [0-xd, b, 1], [a-xd, b, 1]]
    M2 = Mcontsub(getf, p, compoundexp, xlim, pars)
    res2 = append(res2, det(M2))

M3 = Mcontsub(getf, p, uniform, xlim, xlim)

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pl.rcParams.update(params)
pl.figure(1)
pl.clf()
ymarg = 0.160
xmarg = 0.2
pl.axes([ymarg,xmarg,0.95-ymarg,0.95-xmarg])
pl.plot(xdif, (det(M3)/res)**(1/3), 'k-', label='Optimal')
pl.plot(xdif, (det(M3)/res2)**(1/3), 'r:', label='Sub-optimal')
pl.plot([xdif[0], xdif[-1]], [1, 1], 'b--', label=r'Uniform')
pl.ylim([0.5, 1.1])
pl.xlim([xdif[0], xdif[-1]])
pl.xlabel(r'$\Delta x_0$')
pl.ylabel(r'$D_\mathrm{eff}$')
pl.legend(loc='upper left')
pl.savefig('fig3.eps')






############# Figure: D_eff vs G'

def uniform(x, pars):
    prob = 1 / abs(pars[1] - pars[0])
    return prob

# M3 = Mcontsub(getf, p, uniform, xlim, xlim)
# M3c = det(M3)/det(M3[1:,1:])
# print (M3c/M1c)


p = [0, 1, 1]
xi = array([[-0.775, 0, 0.775], [1/3, 1/3, 1/3]])
xlim = [-3, 3]

gdif = linspace(0.4, 2.2, 61)
res = array([])
for gd in gdif:
    xi = array([[-0.775/gd, 0, 0.775/gd], [1/3, 1/3, 1/3]])
    M1 = Mcontinuous(getf, p, xi)
    res = append(res, det(M1))

def compoundexp(x, pars):
    """
    e.g. pars = ((-a, b, 1), (a, b, 1))
    """
    h = 0
    for par in pars:
        try:
            prob += par[2]*exp(-(x-par[0])**2/(2*par[1]**2))/sqrt(2*pi*par[1])
        except NameError:
            prob = par[2]*exp(-(x-par[0])**2/(2*par[1]**2))/sqrt(2*pi*par[1])
        h += par[2]
    prob = prob /h
    return prob

res2 = array([])
for gd in gdif:
    a = 0.775
    b = 0.4
    pars = [[-a/gd, b/gd, 1], [0, b/gd, 1], [a/gd, b/gd, 1]]
    M2 = Mcontsub(getf, p, compoundexp, xlim, pars)
    res2 = append(res2, det(M2))

M3 = Mcontsub(getf, p, uniform, xlim, xlim)

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pl.rcParams.update(params)
pl.figure(1)
pl.clf()
ymarg = 0.160
xmarg = 0.2
pl.axes([ymarg,xmarg,0.95-ymarg,0.95-xmarg])
pl.plot(gdif, (det(M3)/res)**(1/3), 'k-', label='Optimal')
pl.plot(gdif, (det(M3)/res2)**(1/3), 'r:', label='Sub-optimal')
pl.plot([gdif[0], gdif[-1]], [1, 1], 'b--', label=r'Uniform')
pl.ylim([0.5, 1.1])
pl.xlim([gdif[0], gdif[-1]])
pl.xlabel(r"$\Gamma'$")
pl.ylabel(r'$D_\mathrm{eff}$')
# pl.legend(loc='lower right')
# pl.show()
pl.savefig('fig4.eps')




############# Figure: D_s optimum D_eff vs dx0

def uniform(x, pars):
    prob = 1 / abs(pars[1] - pars[0])
    return prob

p = [0, 1, 1]
c = 0.05
xlim = [-2, 2]

xdif = linspace(0, 0.1, 61)
res = array([])
for xd in xdif:
    w = 0.001
    xi = array([[-0.576-xd, 0-xd, 0.576-xd], [1/2-w, w/2, 1/2-w]])
    # xi = array([[-0.576-xd, 0.576-xd], [1/2, 1/2]])
    M1 = Mcontinuous(getf, p, xi)
    res = append(res, det(M1)/det(M1[1:,1:]))

def compoundexp(x, pars):
    """
    e.g. pars = ((-a, b, 1), (a, b, 1))
    """
    h = 0
    for par in pars:
        try:
            prob += par[2]*exp(-(x-par[0])**2/(2*par[1]**2))/sqrt(2*pi*par[1])
        except NameError:
            prob = par[2]*exp(-(x-par[0])**2/(2*par[1]**2))/sqrt(2*pi*par[1])
        h += par[2]
    prob = prob /h
    return prob

res2 = array([])
for xd in xdif:
    a = 0.576
    b = 0.15
    pars = [[-a-xd, b, 1], [a-xd, b, 1]]
    M2 = Mcontsub(getf, p, compoundexp, xlim, pars)
    res2 = append(res2, det(M2)/det(M2[1:,1:]))

M3 = Mcontsub(getf, p, uniform, xlim, xlim)
dM3 = det(M3)/det(M3[1:,1:])

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pl.rcParams.update(params)
pl.figure(1)
pl.clf()
ymarg = 0.160
xmarg = 0.2
pl.axes([ymarg,xmarg,0.95-ymarg,0.95-xmarg])
pl.plot(xdif, (dM3/res), 'k-', label='Optimal')
pl.plot(xdif, (dM3/res2), 'r:', label='Sub-optimal')
pl.plot([xdif[0], xdif[-1]], [1, 1], 'b--', label=r'Uniform')
pl.ylim([0, 1.1])
pl.xlim([xdif[0], xdif[-1]])
pl.xlabel(r'$\Delta x_0$')
pl.ylabel(r'$D_\mathrm{eff}$')
pl.legend(loc='lower right')
# pl.show()
pl.savefig('fig5.eps')


# ############# Figure: Lorentz vs Gauss, design

lorentz = lambda p, x: p[2] * p[1] / ((x-p[0])**2 + p[1]**2)
gauss = lambda p, x: p[2] / p[1] * exp( - (x-p[0])**2/(2*p[1]**2))

p = 0, 1, 1
xc = linspace(-3, 3, 301)



def dofit(f, x, y, p0):
    data = Data(x, y)
    model = Model(f)
    fit = ODR(data, model, p0)
    fit.set_job(fit_type=2)
    output = fit.run()
    return output

pg = dofit(gauss, xc, lorentz(p, xc), p).beta
print pg

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pl.rcParams.update(params)
pl.figure(1)
pl.clf()
ymarg = 0.160
xmarg = 0.2
pl.axes([ymarg,xmarg,0.95-ymarg,0.95-xmarg])


pl.plot(xc, lorentz(p, xc), 'b-', linewidth=2)
pl.plot(xc, gauss(pg, xc), 'g--', linewidth=2)
xp = [-2.73, -0.96, 0, 0.96, 2.73]
for xs in xp:
    pl.plot([xs, xs], [0, max(lorentz(p,xs), gauss(pg, xs))], 'k-')

pl.xlabel(r'$x$')
pl.ylabel(r'$\eta(x)$')
# pl.legend(loc='lower right')
# pl.show()
pl.savefig('fig6.eps')


# ############# Figure: Lorentz vs Gauss, residual

lorentz = lambda p, x: p[2] * p[1] / ((x-p[0])**2 + p[1]**2)
gauss = lambda p, x: p[2] / p[1] * exp( - (x-p[0])**2/(2*p[1]**2))

p = 0, 1, 1
xc = linspace(-3, 3, 301)

def dofit(f, x, y, p0):
    data = Data(x, y)
    model = Model(f)
    fit = ODR(data, model, p0)
    fit.set_job(fit_type=2)
    output = fit.run()
    return output

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pl.rcParams.update(params)
pl.figure(1)
pl.clf()
ymarg = 0.160
xmarg = 0.2
pl.axes([ymarg,xmarg,0.95-ymarg,0.95-xmarg])
xold = loadtxt('xout.txt')
xold = xold[16:]
pg = dofit(gauss, xold, lorentz(p, xold), p).beta
pl.plot(xc, (lorentz(p, xc)-gauss(pg, xc))**2*1e3, 'k-')
pl.plot([-3, 3], [7.4, 7.4], 'b--')
pl.plot(xold, (lorentz(p, xold)-gauss(pg, xold))**2*1e3,  'rx', markersize=5)
pl.xlabel(r'$x$')
pl.ylabel(r'$\psi_2(x, \xi_N) (\times 10^{-3})$')
# pl.legend(loc='lower right')
# pl.show()
pl.savefig('fig7.eps')



############# Figure: Lorentz vs Gauss / efficiency

lorentz = lambda p, x: p[2] * p[1] / ((x-p[0])**2 + p[1]**2)
gauss = lambda p, x: p[2] / p[1] * exp( - (x-p[0])**2/(2*p[1]**2))

p = 0, 1, 1
xc = linspace(-3, 3, 301)

def dofit(f, x, y, p0):
    data = Data(x, y)
    model = Model(f)
    fit = ODR(data, model, p0)
    fit.set_job(fit_type=2)
    output = fit.run()
    return output

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 10,
          'text.fontsize': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size}
pl.rcParams.update(params)
pl.figure(1)
pl.clf()
ymarg = 0.160
xmarg = 0.2
pl.axes([ymarg,xmarg,0.95-ymarg,0.92-xmarg])
xold = loadtxt('xout.txt')
ntest = range(5, 100, 1)
res = array([])
for n in ntest:
    xt = xold[:n]
    pg = dofit(gauss, xt, lorentz(p, xt), p).beta
    res = append(res, sum((lorentz(p, xt)-gauss(pg, xt))**2)/len(xt))
pl.plot(ntest, res/7.32e-3, '.')
pl.ylim([0.6, 1])
pl.xlabel(r'$N$')
pl.ylabel(r'$\Delta_2(\xi_N) / \Delta_2(\xi^*)$')
# pl.show()
pl.savefig('fig8.eps')






