from __future__ import division
from numpy import *
from numpy.linalg import inv
import pylab as pl

##### Optimizing when there are three parameters for Lorentzian

# P = [x0, G, A, B]
dx0 = lambda p, x: 2*(x-p[0])*p[2]/((x-p[0])**2 + p[1]**2)**2
dg = lambda p, x: -2*p[1]*p[2]/((x-p[0])**2 + p[1]**2)**2
da = lambda p, x: 1/((x-p[0])**2 + p[1]**2)

def getf(p, x):
    return array([[dx0(p, x)],[dg(p,x)], [da(p, x)]])

p = [0, 1, 1]


x = array([-0.6, -0.5, 0, 0.5, 0.6])
def getnewx(x):
    n = len(x)
    F = zeros((n, len(p)))
    for i, xi in enumerate(x):
        F[i] = getf(p, xi).T[0]
    M = dot(F.T, F)/n
    dfunc = lambda x: dot(getf(p, x).T, dot(inv(M), getf(p, x)))
    x2 = linspace(-3, 3, 301)
    dx2 = []
    for xt in x2:
        dx2.append(dfunc(xt)[0][0])
    return x2[argmax(dx2)]

n = 70
for i in xrange(n-len(x)):
    newx = getnewx(x)
    x = append(x, newx)
    # x = append(x, -newx)

x = x[10:]
print x
pl.hist(x)
pl.show()
