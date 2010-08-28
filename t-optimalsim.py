from __future__ import division
from numpy import *
from scipy.linalg import inv, det
import pylab as pl


n1 = lambda p, x: p[0] + p[1]*exp(x) + p[2]*exp(-x)
n2 = lambda p, x: p[0] + p[1]*x + p[2]*x**2
# n1 = lambda p, x: p[1]*exp(x) + p[2]*exp(-x)
# n2 = lambda p, x: p[0]*x + p[1]*x**2

xc = linspace(-1, 1, 101)



dl11 = lambda x: x*0 + 1
dl12 = lambda x: exp(x)
dl13 = lambda x: exp(-x)

dl21 = lambda x: x*0 + 1
dl22 = lambda x: x
dl23 = lambda x: x**2



def getf1(x):
    return array([[dl11(x)], [dl12(x)], [dl13(x)]])
    # return array([[dl12(x)], [dl13(x)]])

def getf2(x):
    return array([[dl21(x)], [dl22(x)], [dl23(x)]])
    # return array([[dl22(x)], [dl23(x)]])



xa = array([[-1, -0.669, 0.144, 0.987], 
            [0.253, 0.428, 0.247, 0.072]])

# #### Include the distribution
def getest(f1, f2, xa):
    x = xa[0,:]
    for i, xi in enumerate(x):
        try:
            M1 += dot(f2(xi), f1(xi).T)*xa[1, i]
        except:
            M1 = dot(f2(xi), f1(xi).T)*xa[1, i]
        try:
            M2 += dot(f2(xi), f2(xi).T)*xa[1, i]
        except:
            M2 = dot(f2(xi), f2(xi).T)*xa[1, i]
    return dot(inv(M2), M1)

def getdelta(ft, f2, pt, p2, xa):
    res = 0
    for i, x in enumerate(xa[0,:]):
        res += (ft(pt, x) - f2(p2, x))**2 * xa[1,i]
    return res

# def iterative(f1, ff1, f2, ff2, p1, xi, maxiter):

#     x = xi
#     xl = linspace(-1, 1, 201)
#     for i in range(maxiter-len(xi)):
#         FF = geteste(ff1, ff2, x)
#         p2 = dot(FF, array(p1))
#         psi = [(f1(p1, xv) - f2(p2, xv))**2 for xv in xl]
#         m = argmax(psi)
#         x = append(x, xl[m])
#         print "%d : %f (%f)" %(i, xl[m], psi[m])
#     return x

# def geteste(f1, f2, xa):
#     x = xa
#     for i, xi in enumerate(x):
#         try:
#             M1 += dot(f2(xi), f1(xi).T)/N
#         except:
#             M1 = dot(f2(xi), f1(xi).T)/N
#         try:
#             M2 += dot(f2(xi), f2(xi).T)/N
#         except:
#             M2 = dot(f2(xi), f2(xi).T)/N
#     return dot(inv(M2), M1)

# def geteste2(f1, f2, xa):
#     x = xa
#     N = len(xa)
#     for i, xi in enumerate(x):
#         # F2 F1
#         try:
#             M21 += dot(f2(xi), f1(xi).T)
#         except:
#             M21 = dot(f2(xi), f1(xi).T)

#         # F2 F2
#         try:
#             M22 += dot(f2(xi), f2(xi).T)
#         except:
#             M22 = dot(f2(xi), f2(xi).T)

#         # F1 F2
#         try:
#             M12 += dot(f1(xi), f2(xi).T)
#         except:
#             M12 = dot(f1(xi), f2(xi).T)

#         # F1 F1
#         try:
#             M11 += dot(f1(xi), f1(xi).T)
#         except:
#             M11 = dot(f1(xi), f1(xi).T)

#     M = M11 - dot(M12, dot(inv(M22), M21))
#     return M

# def iterative2(f1, ff1, f2, ff2, p1, xi, maxiter):

#     x = xi
#     xl = linspace(-1, 1, 201)
#     for i in range(maxiter-len(xi)):
#         FF = geteste2(ff1, ff2, x)
#         p2 = dot(FF, array(p1))
#         psi = [(f1(p1, xv) - f2(p2, xv))**2 for xv in xl]
#         m = argmax(psi)
#         x = append(x, xl[m])
#         print "%d : %f (%f)" %(i, xl[m], psi[m])
#     return x


def iterative(f1, ff1, f2, ff2, p1, x0, maxiter, s):
    xt = linspace(-1, 1, 81)
    x = x0
    nc = zeros(len(x))
    for i in range(maxiter-len(x0)):
        y1 = n1(p1, x) + random.randn(len(x))*s
        F1 = zeros((len(x), 3))
        F2 = zeros((len(x), 3))
        for i, xi in enumerate(x):
            F1[i, :] = ff1(xi).T[:]
            F2[i, :] = ff2(xi).T[:]
        p1e = dot(inv(dot(F1.T, F1)),dot(F1.T, y1))
        p2e = dot(inv(dot(F2.T, F2)),dot(F2.T, y1))
        psi = [(f1(p1e, xi) - f2(p2e, xi))**2 for xi in xt]
        m = argmax(psi)
        x = append(x, xt[m])
        nc = append(nc, sum(psi))
    return x, nc

p1 = [4.5, -1.5, -2]
s = 0.1
x0 = array([-1, 0, 1])
maxiter = 400


# xa = array([[-1, -0.869, 0.184, 0.987], 
#             [0.253, 0.428, 0.247, 0.072]])
# # xa = array([[-1, -0.5, 0.5, 0.8], 
# #             [1/4, 1/4, 1/4, 1/4]])
# FF = getest(getf1, getf2, xa)
# p1e = [4.5, -1.5, -2]
# p2e = dot(FF, array(p1e))
# print p2e
# print getdelta(n1, n2, p1e, p2e, xa)

#### DO & Save
xout, nc = iterative(n1, getf1, n2, getf2, p1, x0, maxiter, s)
# savetxt('xout.txt', xout)
#### Reload
# xout = loadtxt('xout.txt')
####
# print xout
pl.figure(3)
pl.hist(xout, 81)
# pl.show()

# xa = array([[-1, -0.5, 0.5, 0.8], 
#             [20, 20, 20, 20]])
# xout = array([])
# for i, xaa in enumerate(xa[0,:]):
#     xout = append(xout, array([xaa]*xa[1,i]))

# xout = array([-0.8, -0.4, -0.3, 0.4, 0.6, 0.7])
x = xout
y1 = n1(p1, x) + random.randn(len(x))*s
F1 = zeros((len(x), 3))
F2 = zeros((len(x), 3))
for i, xi in enumerate(x):
    F1[i, :] = getf1(xi).T[:]
    F2[i, :] = getf2(xi).T[:]
p1e = dot(inv(dot(F1.T, F1)),dot(F1.T, y1))
p2e = dot(inv(dot(F2.T, F2)),dot(F2.T, y1))
RSS1 = sum((y1 - n1(p1e, x))**2)
RSS2 = sum((y1 - n2(p2e, x))**2)
RSSx = sum((y1 - n1(p1, x))**2)
pl.figure(15)
pl.plot(x, y1-n1(p1, x), 'x')
pl.plot(x, y1-n1(p1e, x), 'o')
# print "Model1 RSS: ", RSS1
# print "Model2 RSS: ", RSS2
print "Model1 RSS: ", RSS1, RSS1/s**2
print "Model2 RSS: ", RSS2, RSS2/s**2
print "MinRSS: ", s**2*maxiter
from scipy.stats import chi2
print "P1:", 1-chi2.cdf(RSS1/(s**2), maxiter-3-1)
print "P2:", 1-chi2.cdf(RSS2/(s**2), maxiter-3-1)

# pl.figure(10)
# pl.hist((y1-n1(p1e,x)))
# pl.hist((y1-n2(p2e,x)))

pl.figure(1)
pl.plot(x, y1, '.')
pl.plot(xc, n1(p1, xc), 'k--')
pl.plot(xc, n1(p1e, xc), 'b')
pl.plot(xc, n2(p2e, xc), 'r')

pl.figure(6)
dd = lambda x: (n1(p1e, x) - n2(p2e, x))**2
print "Test:", sum(dd(xout)/len(xout))
pl.plot(xc, dd(xc))
pl.plot(xout, dd(xout), 'x')
# pl.plot(x, y1 - n1(p1e,x), 'o')
# pl.plot(x, y1 - n2(p2e,x), 'x')

# pl.figure(2)
# pl.plot(nc)

#############
# d2 = array([])
# nx = maxiter
# for i in xrange(4,nx):
#     xv = xout[:(i+1)]
#     psi = sum([(n1(p1, x) - n2(p2e, x))**2/len(xv) for x in xv])
#     d2 = append(d2, psi)
# pl.figure(4)
# pl.plot(d2/1.087e-3, '.')
# # pl.ylim([0, 1])
# pl.show()

print p1
print p1e
print p2e
psi = sum([(n1(p1e, xi) - n2(p2e, xi))**2 for xi in xout])/len(xout)
print psi
delta = 1.087e-3
xl = linspace(-1, 1, 201)
psi = [(n1(p1e, xi) - n2(p2e, xi))**2 for xi in xl]
pl.figure(4)
pl.plot(xl, psi)
pl.plot([xl[0], xl[-1]], [delta, delta], '--')
for x in xout:
    pl.plot(x, delta, 'k.')
pl.show()


# pl.plot(xc, n1(p1e, xc))
# pl.plot(xc, n2(p2e, xc))
# pl.show()


# x = xa[0,:]
# FF =  getest(getf1, getf2, xa)
# p1 = [4.5, -0.5, -2]
# p2 = dot(FF, array(p1))
# # p2 = dot(FF, array(p1[1:]))
# print p2

# # p2[0] += 0
# # pl.plot(xc, n1(p1, xc))
# # pl.plot(xc, n2(p2, xc))
# # pl.show()
# xin = array([-1, -0.25, 0.25, 1])
# xout = iterative(n1, getf1, n2, getf2, p1, xin, 500)
# # pl.hist(xout)
# # pl.show()

# delta = 0
# for i, xi in enumerate(xout):
#     delta += (n1(p1, xi) - n2(p2, xi))**2
# delta /= len(xout)
# print delta

# FF =  geteste(getf1, getf2, xout)
# p2 = dot(FF, array(p1))

# # delta = 0
# # for i, xi in enumerate(x):
# #     delta += (n1(p1, xi) - n2(p2, xi))**2 * xa[1, i]

# # print delta

# xl = linspace(-1, 1, 201)
# psi = [(n1(p1, xi) - n2(p2, xi))**2 for xi in xl]
# pl.plot(xl, psi)
# pl.plot([xl[0], xl[-1]], [delta, delta], '--')
# pl.show()
