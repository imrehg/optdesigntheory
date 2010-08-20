from __future__ import division
from numpy import *
from numpy.linalg import inv
import pylab as pl

# dx = lambda x, t: x**2 * exp(-2*t*x)/exp(-2*t**2)
# dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)

# dx = lambda x, x0, g : dlx0(x, x0, g)**2/dlx0(g/5, x0, g)**2

# x = linspace(-2,2, 301)

# pl.plot(x, dx(x, 0, 1))
# pl.show()

dx0 = lambda p, x: 2*(x-p[0])/((x-p[0])**2 + p[1]**2)**2
dg = lambda p, x: -2*p[1]/((x-p[0])**2 + p[1]**2)**2

dx0 = lambda p, x: 2*(x-p[0])/((x-p[0])**2 + p[1]**2)**2
dg = lambda p, x: -2*p[1]/((x-p[0])**2 + p[1]**2)**2

def getf(p, x):
    return array([[dx0(p, x)],[dg(p,x)]])

p = [0, 1]

# M = dot(getf(p, x),getf(p, x).T)
# print M
# print inv(M)
n = 101

# x = linspace(-3, 3, n)*0-0.5
# x = [-0.5, -0.42, -0.39, 0, 0.39, 0.42, 0.5]


# for i, xi in enumerate(x):
#     F[i] = getf(p, xi).T[0]
# M = dot(F.T, F)/n

# dfunc = lambda x: dot(getf(p, x).T, dot(inv(M), getf(p, x)))
# x2 = linspace(-2, -0.1, 101)
# dx2 = []
# for xt in x2:
#     dx2.append(dfunc(xt)[0][0])
# dx2 = array(dx2)

# pl.plot(x2, dx2)
# pl.show()

x = array([-0.5, 0, 0.5])
def getnewx(x):
    n = len(x)
    F = zeros((n, 2))
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
for i in xrange(n-3):
    newx = getnewx(x)
    x = append(x, newx)
    # x = append(x, -newx)

x = x[10:]
print x
pl.hist(x)
pl.show()
