###
# Optimal and robust sub-optimal design code
# Atkinson & Donev, Optimal Experimental Designs
###

from __future__ import division
from numpy import *
from scipy.odr import *
from scipy.linalg import det, inv
from scipy.optimize import fmin
from scipy.integrate import quad


def Mcontinuous(f, p, xi):
    """
    The information matrix for a given continuous design

    Input:
    f : the function to calculate the information matrix elements
        in the form of f(p, x)
    p : imput parameters for f
    xi : continuous design - [[x1, x2, ..., xn], [w1, w2, ..., wn]]
         of points xi and weights wi, such that Sum(wi) = 1

    Atkinson Eq. 9.6, p95
    """
    for i in xrange(len(xi[0,:])):
        x = xi[0, i]
        w = xi[1, i]
        fx = f(p, x)
        try:
            M += dot(fx, fx.T)*w
        except:
            M = dot(fx, fx.T)*w
    return M

def Mexact(f, p, xi):
    """
    The information matrix for a given exact design

    Input:
    f : the function to calculate the information matrix elements
        in the form of f(p, x)
    p : imput parameters for f
    xi : measurement points

    Atkinson p95
    """
    for x in xi:
        fx = f(p, x)
        try:
            M += dot(fx, fx.T)
        except:
            M = dot(fx, fx.T)
    M /= len(xi)
    return M

def Mcontsub(f, p, xi, xlim, xiparms):
    """
    Suboptimal continuous M

    Input:
    f(p, x) : information matrix row
    p : input parameters
    xi(x, xparms) : probability distribution for xi
    xlim : integration limits
    xiparms : parameters for the probability distribution
    """
    norm = quad(xi, xlim[0], xlim[1], args=(xiparms))[0]
    n = len(f(p, xlim[0]))
    M = zeros((n,n))
    for fi in xrange(n):
        for fj in xrange(n):
            infointeg = lambda x, p, xiparms, i, j: f(p, x)[i]*f(p, x)[j]*xi(x, xiparms)
            M[fi, fj] = quad(infointeg, xlim[0], xlim[1], 
                             args=(p, xiparms, fi, fj))[0] / norm
    return M

def sdvarcont(f, p, xi, x, optimize=False):
    """
    Standardized variance for continuous design
    """
    n = len(f(p, xi[0]))
    e = 1e-5
    M = Mcontinuous(f, p, xi)-diag([e]*n)
    Minv = inv(M)
    ret = array([])
    if optimize:
        Minv2 = inv(M[1:,1:])
    for pos in x:
        ret = append(ret, dot(f(p, pos).T, dot(Minv, f(p, pos))))
        if optimize:
            ret[-1] -= dot(f(p, pos)[1:].T, dot(Minv2, f(p, pos)[1:]))
    return ret

def sdvarexact(f, p, xi, x, optimize=False):
    """
    Standardized variance for exact design
    """
    # guarding agains singular matrices
    n = len(f(p, xi[0]))
    e = 1e-7
    M = Mexact(f, p, xi)-diag([e]*n)
    Minv = inv(M)
    if optimize:
        Minv2 = inv(M[1:,1:])
    ret = array([])
    for pos in x:
        ret = append(ret, dot(f(p, pos).T, dot(Minv, f(p, pos))))
        if optimize:
            ret[-1] -= dot(f(p, pos)[1:].T, dot(Minv2, f(p, pos)[1:]))
    return ret

def sequential(f, p, xstart, xlim, nmax, optimize=False):
    """
    Sequential generation N-point design
    """
    x = xstart
    xl = linspace(xlim[0], xlim[1], 2001)
    for i in xrange(len(x)+1, nmax):
        dxk = sdvarexact(f, p, x, xl, optimize)
        m = argmax(abs(dxk))
        # print dxk[m]
        x = append(x, xl[m])
    return x
