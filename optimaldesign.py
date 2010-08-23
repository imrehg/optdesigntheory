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

def sdvarcont(f, p, xi, x):
    """
    Standardized variance for continuous design
    """
    Minv = inv(Mcontinuous(f, p, xi))
    ret = array([])
    for pos in x:
        ret = append(ret, dot(f(p, pos).T, dot(Minv, f(p, pos))))
    return ret

def sdvarexact(f, p, xi, x):
    """
    Standardized variance for exact design
    """
    Minv = inv(Mexact(f, p, xi))
    ret = array([])
    for pos in x:
        ret = append(ret, dot(f(p, pos).T, dot(Minv, f(p, pos))))
    return ret

def sequential(f, p, xstart, xlim, nmax):
    """
    Sequential generation N-point design
    """
    x = xstart
    xl = linspace(xlim[0], xlim[1], 1001)
    for i in xrange(len(x)+1, nmax):
        dxk = sdvarexact(f, p, x, xl)
        m = argmax(dxk)
        x = append(x, xl[m])
    return x


        
