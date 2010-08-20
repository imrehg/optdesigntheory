from __future__ import division
from numpy import *
from scipy.odr import *
from scipy.linalg import det, inv
from scipy.optimize import fmin
from scipy.integrate import quad
import pylab as pl

lorentz = lambda x, x0, g, A: A/((x-x0)**2+g**2)
lmodel = lambda beta, x: lorentz(x, beta[0], beta[1], beta[2])

x0 = 0
g = 1
A = 1

def x_uniform1(xlim, n):
    """ Uniform steps within the limits """
    return linspace(xlim[0], xlim[1], n)

xlim = [-4, 4]
n2 = 33
n = 330
rep = 400
b0 = [x0, g, A]
sd = 1/50

##############
x = x_uniform1(xlim, n)

def compoundexp(pars, x):
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
    prob = prob / h
    return prob


##############################
## Theoretical calculation of |M| = |F^T*F|
##############################

dlx0 = lambda p, x: (2*(x-p[0])*p[2]/((x-p[0])**2+p[1]**2)**2)
dlg = lambda p, x: (-2*p[2]*p[1])/((x-p[0])**2+p[1]**2)**2
dla = lambda p, x: zeros(shape(p[2]))+1/((x-p[0])**2+p[1]**2)

def getf(p, x):
    return array([[dlx0(p, x)],[dlg(p,x)],[dla(p,x)]])

# Distribution parameters
p = [0, 1, 1]

def infointeguni(x, p, xlim, ij):
    gf = getf(p, x)
    return gf[ij[0]]*gf[ij[1]]/diff(xlim)

def infointeg(x, p, pars, ij):
    gf = getf(p, x)
    return gf[ij[0]]*gf[ij[1]]*compoundexp(pars, x)
# x1 = infointeg(x, p, pars, [0, 0])
# print quad(infointeg, xlim[0], xlim[1], args=(p, pars, [1, 1]))[0]

F = zeros((3,3))
for fi in xrange(3):
    for fj in xrange(3):
        F[fi, fj] = quad(infointeguni, xlim[0], xlim[1], args=(p, xlim, [fi, fj]))[0]
unidet = det(dot(F.T,F))
print unidet

### Grid with (ng, ng) points for search
ng = 10
al = linspace(0, 2, ng)
bl = linspace(0.01, 2, ng)
X, Y = meshgrid(al, bl)
Z = zeros((ng, ng))
for i in xrange(ng):
    print "i#%d" %(i)
    for j in xrange(ng):
        a = X[i,j]
        b = Y[i,j]
        # pars0 = ((-a, b, 1), (a, b, 1))
        pars0 = ((-a, b, 1), (0, b, 1), (a, b, 1))
        F = zeros((3,3))
        for fi in xrange(3):
            for fj in xrange(3):
                F[fi, fj] = quad(infointeg, xlim[0], xlim[1], args=(p, pars0, [fi, fj]))[0]
        Z[i, j] = det(dot(F.T,F))/unidet
                
CS = pl.contour(X, Y, Z, 27)
pl.clabel(CS, inline=1, fontsize=10)
pl.xlabel("Exponential offset")
pl.ylabel("Exponential width")
pl.show()
#############################

