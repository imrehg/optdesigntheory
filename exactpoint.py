from __future__ import division
from numpy import *
from numpy.linalg import inv, det
import pylab as pl

##### Optimizing when there are three parameters for Lorentzian

# P = [x0, G, A, B]
dx0 = lambda p, x: (2*(x-p[0])*p[2]/((x-p[0])**2+p[1]**2)**2)
dg = lambda p, x: (-2*p[2]*p[1])/((x-p[0])**2+p[1]**2)**2
da = lambda p, x: zeros(shape(p[2]))+1/((x-p[0])**2+p[1]**2)

def getf(p, x):
    # return array([[dx0(p, x)], [dg(p, x)], [da(p, x)]])
    return array([[dg(p, x)], [dx0(p, x)], [da(p, x)]])
    # return array([[dx0(p, x)], [dg(p, x)], [da(p, x)], [2*dx0(p, x)*dg(p, x)], [2*dx0(p, x)*da(p, x)], [2*dg(p, x)*da(p, x)]])
    # return array([[dg(p, x)], [dx0(p, x)], [da(p, x)]])
    # return array([[da(p, x)], [dx0(p, x)], [dg(p, x)]])

p = [0, 1, 1]

def getM(x):
    n = len(x)
    F = zeros((n, 3))
    for i, xi in enumerate(x):
        F[i] = getf(p, xi).T[0]

    F = matrix(F)
    M = dot(F.T, F)
    e = 1e-5
    M = M + diag([e]*3)
    # return det(M)
    M22 = dot(F[1:,1:].T,F[1:,1:])
    M22 = M22 + diag([e]*2)
    return det(M) / det(M22)

    # M = dot(F.T, F)/n
    # M22 = dot(F[1:,1:].T,F[1:,1:])/n

    # dfunc = lambda x: dot(getf(p, x).T, dot(inv(M), getf(p, x)))
    # # dfunc = lambda x: dot(getf(p, x).T, dot(inv(M), getf(p, x))) - dot(getf(p, x)[1:].T, dot(inv(M22), getf(p, x)[1:]))
    # #dfunc = lambda x: dot(getf(p, x).T, dot(inv(M), getf(p, x))) - dot(getf(p, x)[1:].T, dot(inv(M[1:,1:]), getf(p, x)[1:]))
    # x2 = linspace(-3, 3, 301)
    # dx2 = []
    # for xt in x2:
    #     dx2.append(dfunc(xt)[0][0])
    # return x2[argmax(dx2)]

aspace = linspace(0.1, 2, 3001)
ret = array([])
for a in aspace:
    ret = append(ret, getM([-a, 0, a]))
mm = argmax(ret)
print "Maximum near: %g => %g" %(aspace[mm], ret[mm])
pl.plot(aspace, ret)
pl.plot([aspace[mm], aspace[mm]], [0, ret[mm]])
pl.show()
# print getnewx(x)

# n = 120
# for i in xrange(n-len(x)):
#     newx = getnewx(x)
#     x = append(x, newx)
#     # x = append(x, -newx)

# x = x[-9:]
# print x
# pl.hist(x)
# pl.show()
