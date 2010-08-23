from __future__ import division
from numpy import *
from scipy.odr import *
from scipy.linalg import det, inv
from scipy.optimize import fmin
from scipy.integrate import quad
import pylab as pl

######
# Calculate d(x, khi) = f(x).T M^-1 f(x)

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
    prob = prob / h
    return prob


##############################
## Theoretical calculation of |M| = |F^T*F|
##############################

dlx0 = lambda p, x: (2*(x-p[0])*p[2]/((x-p[0])**2+p[1]**2)**2)
dlg = lambda p, x: (-2*p[2]*p[1])/((x-p[0])**2+p[1]**2)**2
dla = lambda p, x: zeros(shape(p[2]))+1/((x-p[0])**2+p[1]**2)

def getf(p, x):
    # return array([[dlx0(p, x)],[dlg(p,x)],[dla(p,x)]])
    return array([[dlg(p, x)],[dlx0(p,x)],[dla(p,x)]])
    # return array([[dla(p, x)],[dlx0(p,x)],[dlg(p,x)]])

# Distribution parameters
p = [0, 1, 1]

def infointeguni(x, p, xlim, ij):
    gf = getf(p, x)
    return gf[ij[0]]*gf[ij[1]]/diff(xlim)

def infointeg(x, p, pars, ij):
    gf = getf(p, x)
    return gf[ij[0]]*gf[ij[1]]*compoundexp(x, pars)
# x1 = infointeg(x, p, pars, [0, 0])
# print quad(infointeg, xlim[0], xlim[1], args=(p, pars, [1, 1]))[0]


F = zeros((3,3))
for fi in xrange(3):
    for fj in xrange(3):
        F[fi, fj] = quad(infointeguni, xlim[0], xlim[1], args=(p, xlim, [fi, fj]))[0]
Minvuni = inv(dot(F.T,F))
x = linspace(xlim[0], xlim[1], 301)
ret = array([])
for pos in x:
    ret = append(ret, dot(getf(p, pos).T, dot(Minvuni, getf(p, pos))))
retres = [mean(ret), min(ret), max(ret)]
print "%.1f %.1f %.1f %.1f" %(retres[0], retres[1], retres[2], retres[2]-retres[1])

# pl.plot(x, ret, label="Uni | mean: %.1f, lim: %.1f - %.1f" %(retres[0], retres[1], retres[2]))
# #### Plot single line 
# a = 0
# b = 1.5
# #pars0 = ((-a, b, 1), (0, b, 1), (a, b, 1))
# pars0 = [[-a, b, 1], [a, b, 1]]
# F = zeros((3,3))
# for fi in xrange(3):
#     for fj in xrange(3):
#         # F[fi, fj] = quad(infointeg, xlim[0], xlim[1], args=(p, pars0, [fi, fj]))[0]
#         F[fi, fj] = quad(infointeg, xlim[0], xlim[1], args=(p, pars0, [fi, fj]))[0] /  \
#             quad(compoundexp, xlim[0], xlim[1], args=(pars0))[0]
# Minv = inv(dot(F.T,F))
# detF = det(dot(F.T,F))
# print detF
# x = linspace(xlim[0], xlim[1], 301)
# ret = array([])
# for pos in x:
#     ret = append(ret, dot(getf(p, pos).T, dot(Minv, getf(p, pos))))
# retres = [mean(ret), min(ret), max(ret)]
# pl.plot(x, ret, label="Exp: | mean: %.1f, lim: %.1f - %.1f" %(retres[0], retres[1], retres[2]))
# pl.legend(loc="best")
# pl.show()



### Grid with (ng, ng) points for search
ng = 5
al = linspace(0, 1.5, ng)
bl = linspace(0.02, 1, ng)
X, Y = meshgrid(al, bl)
x = linspace(xlim[0], xlim[1], 301)
Zmean = zeros((ng, ng))
Zmin = zeros((ng, ng))
Zmax = zeros((ng, ng))
Zspread = zeros((ng, ng))
Zinfo = zeros((ng, ng))
for i in xrange(ng):
    print "i#%d" %(i)
    for j in xrange(ng):
        a = X[i,j]
        b = Y[i,j]
        # pars0 = [[-a, b, 1], [a, b, 1]]
        pars0 = [[-a, b, 1], [0, b, 1], [a, b, 1]]
        M = zeros((3,3))
        for fi in xrange(3):
            for fj in xrange(3):
                M[fi, fj] = quad(infointeg, xlim[0], xlim[1], args=(p, pars0, [fi, fj]))[0] / \
                    quad(compoundexp, xlim[0], xlim[1], args=(pars0))[0]

        Minv = inv(M)
        ret = array([])
        for pos in x:
            ret = append(ret, dot(getf(p, pos).T, dot(Minv, getf(p, pos))))

        Minv2 = inv(M[1:,1:])
        for pi, pos in enumerate(x):
            ret[pi] -= dot(getf(p, pos)[1:].T, dot(Minv2, getf(p, pos)[1:]))
        
        Zinfo[i, j] = det(M) / det(M[1:,1:])
        # Zinfo[i, j] = det(M)
        Zmean[i, j] = log(mean(ret))
        Zmin[i, j] = log(min(ret))
        Zmax[i, j] = log(max(ret))
        Zspread[i, j] = log((Zmax[i, j] - Zmin[i, j]))

Zm = [Zinfo, Zmean, Zmax]
Zname = ["D-optimum", "V-optimum", "G-optimum"]
for i,Z in enumerate(Zm):
    pl.figure()
    CS = pl.contour(X, Y, Z, 35)
    pl.clabel(CS, inline=1, fontsize=10)
    pl.xlabel("Exponential offset")
    pl.ylabel("Exponential width")
    pl.title(Zname[i])
pl.show()

#############################

