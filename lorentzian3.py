###
# Optimal design with Lorentzian lineshape
###

from __future__ import division
from numpy import *
import pylab as pl
from optimaldesign import *

dlx0 = lambda p, x: (2*(x-p[0])*p[2]/((x-p[0])**2+p[1]**2)**2)
dlg = lambda p, x: (-2*p[2]*p[1])/((x-p[0])**2+p[1]**2)**2
dla = lambda p, x: x*0+1/((x-p[0])**2+p[1]**2)
dlc = lambda p, x: x*0+1

def getf(p, x):
    return array([[dlx0(p, x)],[dlg(p,x)],[dla(p,x)], [dlc(p, x)]])
    # return array([[dlx0(p, x)],[dlg(p,x)],[dla(p,x)]])
    # return array([[dlg(p, x)],[dlx0(p,x)],[dla(p,x)]])
    # return array([[dla(p, x)],[dlx0(p,x)],[dlg(p,x)]])


p = [0, 1, 1, 0]
a = 0.77
xi = array([[-a, 0, a], [1/3, 1/3, 1/3]])
# xi = array([[0, 1, 2], [1/3, 1/3, 1/3]])

# a = 1
# xi = array([[-a, -a/2, a/2, a], [87/400, 113/400, 113/400, 87/400]])
xi = array([[-2, -0.72, 0, 0.72, 2], [1/8, 1/4, 1/4, 1/4, 1/8]])

xlim = [-3, 3]

def uniform(x, pars):
    prob = 1 / (pars[0] - pars[1])
    return prob



# # M1 = Mcontinuous(getf, p, xi)
# # M3 = Mcontsub(getf, p, uniform, xlim, xlim)
# # print M1
# # print M3
# # d1 = det(M1)
# # d2 = det(M3)
# # print (d2/d1)**(1/4)

# # # x = linspace(-2, 2, 101)
# # # pl.plot(x, sdvarcont(getf, p, xi, x))
# # # pl.show()


# nmax = 4*200
# res = sequential(getf, p, xstart, xlim, nmax, optimize=False)
# res = res[-(4*100):]
# print res
# pl.hist(res,20)
# pl.show()

# x = xi[0]
# xl = linspace(xlim[0], xlim[1], 1001)
# dxk = sdvarexact(getf, p, x, xl, False)
# pl.plot(xl, dxk)
# pl.show()

# M3 = Mcontsub(getf, p, uniform, xlim, xlim)
# print M1
# print M3
# d1 = det(M1)
# d2 = det(M3)

# print (d2/d1)**(1/3)
xstart = array([ -3, -0.72, -0.72, 0, 0,  0.72, 0.72, 3])
xlim = [-3, 3]
x = linspace(xlim[0], xlim[1], 101)
sd = sdvarexact(getf, p, xstart, x)
pl.plot(x, sd)
pl.show()

# a = getf(p, x[0])
# print a
# print dot(a.T, a)



lorentz = lambda p, x: p[2]/((x-p[0])**2 + p[1]**2) + p[3]
# pl.plot(x, lorentz(p, x))
# pl.plot(xi[0,:], lorentz(p, xi[0,:]),'o')
# pl.show()


# x = linspace(-2, 2, 101)
# pl.plot(x, sdvarcont(getf, p, xi, x))
# pl.show()


# xi = [-a, 0, a, 0, -a, a]
# x = linspace(-1, 1, 101)
# pl.plot(x, sdvarexact(getf, p, xi, x))
# pl.show()

# xstart = array([-1, -0.5, 0.5, 1])
# xlim = [-3, 3]
# nmax = 100
# res = sequential(getf, p, xstart, xlim, nmax, optimize=False)
# res = res[-80:]
# print res
# pl.hist(res)
# pl.show()

# print det(Mcontinuous(getf, p, xi))
# print det(Mexact(getf, p, x))

# al = linspace(0.1, 2, 500)
# ret = array([])
# for a in al:
#     # xi = array([[-a, 0, a], [1/3, 1/3, 1/3]])
#     # xi = array([[-a, -a/2, 0, a/2, a], [1/10, 3/10, 2/10, 3/10, 1/10]])
#     ret = append(ret, det(Mcontinuous(getf, p, xi)))

# mm = argmax(ret)
# print "%g => %g" %(al[mm], ret[mm])
# pl.plot(al, ret)
# pl.show()





# ######
# def uniform(x, pars):
#     prob = 1 / (pars[0] - pars[1])
#     return prob

# def singleexp(x, pars):
#     prob = pars[2]*exp(-(x-pars[0])**2/(2*pars[1]**2))/sqrt(2*pi*pars[1]**2)
#     return prob

# def compoundexp(x, pars):
#     """
#     e.g. pars = ((-a, b, 1), (a, b, 1))
#     """
#     h = 0
#     for par in pars:
#         try:
#             prob += par[2]*exp(-(x-par[0])**2/(2*par[1]**2))/sqrt(2*pi*par[1])
#         except NameError:
#             prob = par[2]*exp(-(x-par[0])**2/(2*par[1]**2))/sqrt(2*pi*par[1])
#         h += par[2]
#     prob = prob /h
#     return prob

# xlim = [-4, 4]
# # pars = [0, 0.1, 1]
# # M2 = Mcontsub(getf, p, singleexp, xlim, pars)


# b = 0.15
# pars = [[-a, b, 1], [0, b, 1], [a, b, 1]]

# a = 1.5
# pars = [[-a, b, 1], [-a/2, b, 1], [a/2, b, 1], [a, b, 1]]

# xi = array([[-a, -a/2, a/2, a], [1/4, 1/4, 1/4, 1/4]])
# # M2 = Mcontsub(getf, p, compoundexp, xlim, pars)

# M1 = Mcontinuous(getf, p, xi)
# M3 = Mcontsub(getf, p, uniform, xlim, xlim)
# print M1
# print M3

# # M1 = M1 + diag([1e-6]*4)
# # print M1
# print det(M1)
# print det(M3)
# # s1 = sqrt(linalg.eigh(inv(M1))[0])
# # s3 = sqrt(linalg.eigh(inv(M3))[0])
# # print s1
# # print s3
# # print [x/y for x, y in zip(s1, s3)]
# # print prod(s1)
# # print prod(s3)

# # print det(M2)/det(M1)
# print det(M3)/det(M1)
# x = linspace(xlim[0], xlim[1], 101)
# # pl.plot(x, compoundexp(x, pars))
# # pl.show()


######### X0 robustness
# xn = 21
# x0p = linspace(0, 1, xn)
# res = array([])
# for x0 in x0p:
#     z = x0
#     a = 0.774
#     xi = array([[-a+z, 0+z, a+z], [1/3, 1/3, 1/3]])
#     MX = Mcontinuous(getf, p, xi)
#     res = append(res, det(MX)/det(M3))

# pl.plot(x0p, res)

# x0p = linspace(0, 1, xn)
# res = array([])
# for x0 in x0p:
#     z = x0
#     a = 0.774
#     b = 0.3
#     pars = [[-a+z, b, 1], [0+z, b, 1], [a+z, b, 1]]
#     MX = Mcontsub(getf, p, compoundexp, xlim, pars)
#     res = append(res, det(MX)/det(M3))

# pl.plot(x0p, res)

# pl.plot([x0p[0], x0p[-1]], [1, 1])
##############

# xn = 21
# x0p = linspace(0.3, 2, xn)
# res = array([])
# for x0 in x0p:
#     z = x0
#     a = 0.774
#     xi = array([[-a/x0, 0/x0, a/x0], [1/3, 1/3, 1/3]])
#     MX = Mcontinuous(getf, p, xi)
#     res = append(res, det(MX)/det(M3))

# pl.plot(x0p, res)

# res = array([])
# for x0 in x0p:
#     z = x0
#     a = 0.774
#     b = 0.3
#     pars = [[-a/x0, b, 1], [0/x0, b, 1], [a/x0, b, 1]]
#     MX = Mcontsub(getf, p, compoundexp, xlim, pars)
#     res = append(res, det(MX)/det(M3))

# pl.plot(x0p, res)

# pl.plot([x0p[0], x0p[-1]], [1, 1])

# pl.show()
