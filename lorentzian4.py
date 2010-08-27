###
# Optimal design with Lorentzian lineshape
###

from __future__ import division
from numpy import *
import pylab as pl
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

# # Original function
# x = linspace(-2, 2)
# pl.plot(x, lorentz([0, 1, 1, 0], x))


# # x0 derivative
# x = -0.1
# G = 1
# A = 1.5
# x0l = linspace(-1, 1, 101)
# h = x0l[1]-x0l[0]
# dx0l = x0l[1:]-h/2
# res = array([lorentz([x0, G, A], x) for x0 in x0l])
# resh = diff(res)/h
# dres = array([dlx0([x0, G, A], x) for x0 in x0l])
# pl.figure()
# pl.plot(dx0l, resh, '.')
# pl.plot(x0l, dres)
# pl.title("X0")

# # Gamma derivative
# x = 0
# x0 = 0
# A = 1.5
# gl = linspace(0.5, 10, 101)
# h = gl[1]-gl[0]
# dgl = gl[1:]-h/2
# res = array([lorentz([x0, G, A], x) for G in gl])
# resh = diff(res)/h
# dres = array([dlg([x0, G, A], x) for G in gl])
# pl.figure()
# pl.plot(dgl, resh, '.')
# pl.plot(gl, dres)
# pl.title("Gamma")

# # Amplitude derivative
# x = 0
# x0 = 2
# G = 1.5
# al = linspace(0.5, 10, 101)
# h = al[1]-al[0]
# dal = al[1:]-h/2
# res = array([lorentz([x0, G, A], x) for A in al])
# resh = diff(res)/h
# dres = array([dla([x0, G, A], x) for A in al])
# pl.figure()
# pl.plot(dal, resh, '.')
# pl.plot(al, dres)
# pl.title("Amplitude")

# pl.show()
# ########################

# p = [0, 1, 1]
# x = 0
# n =3
# e = 1e-6
# a = 0.7746
# xi = array([[-a, 0, a], [1/3, 1/3, 1/3]])


# p = [0, 1, 1]
# xstart = [-1, 0.5, 0]
# xlim = [-3, 0]

# nmax = 650
# res = sequential(getf, p, xstart, xlim, nmax, optimize=False)
# # res = res[-(4*100):]
# res = res[-(3*200):]
# print res
# nn, bins, patches = pl.hist(res,21, histtype='stepfilled')
# for i, n in enumerate(nn):
#     if (n > 0):
#         print "%f : %d" %(bins[i], n)
# print "total: %d" %(sum(nn))
# pl.show()

p = [0, 1, 1]
# a = 1
# xi = [-a, -a, a, a, a, -a, -a, 0, a]

# a = 1
# w = 0.499
# xi = array([[-a, 0, a], [w, 1-2*w, w]])
# xi = array([[-a, 0, a], [1/3, 1/3, 1/3]])

xi = array([[-1.2585, -0.3945, 0], [1/3, 1/3, 1/3]])
# x = linspace(-2, 0, 101)
# res = sdvarcont(getf, p, xi, x, optimize=False)
# print max(res)
# pl.plot(x, res)
# pl.show()





p = [0, 1, 1]
# a = 0.7746
# xi = array([[-a, 0, a], [1/3, 1/3, 1/3]])

# xi = array([[-3, -0.748, 0, 0.748, 3], [1/8, 1/4, 1/4, 1/4, 1/8]])


# xi = array([[-0.579, 0.579], [0.5, 0.5]])
xi = array([[-1.2585, -0.3945, 0], [1/3, 1/3, 1/3]])
M1 = Mcontinuous(getf, p, xi)
xlim = [-2, 0]

def uniform(x, pars):
    prob = 1 / abs(pars[1] - pars[0])
    return prob

M3 = Mcontsub(getf, p, uniform, xlim, xlim)
print (det(M3)/det(M1))**(1/3)










# Width

# xi = array([[-1.188, 0, 1.188], [0.353, 0.294, 0.353]])

# #### height
# a = 1
# w = 0.499
# xi = array([[-a, 0, a], [w, 1-2*w, w]])



# M1 = Mcontinuous(getf, p, xi)
# M1c = det(M1)/det(M1[1:,1:])
# # print det(M1)



# xlim = [-2, 2]

# def uniform(x, pars):
#     prob = 1 / abs(pars[1] - pars[0])
#     return prob

# M3 = Mcontsub(getf, p, uniform, xlim, xlim)
# M3c = det(M3)/det(M3[1:,1:])
# print (M3c/M1c)


# p = [0, 1, 1, 0]
# # xi = array([[0, 1, 2], [1/3, 1/3, 1/3]])

# # a = 1
# # xi = array([[-a, -a/2, a/2, a], [87/400, 113/400, 113/400, 87/400]])
# xi = array([[-2, -0.72, 0, 0.72, 2], [1/8, 1/4, 1/4, 1/4, 1/8]])

# xlim = [-3, 3]

# def uniform(x, pars):
#     prob = 1 / (pars[0] - pars[1])
#     return prob



# # # M1 = Mcontinuous(getf, p, xi)
# # # M3 = Mcontsub(getf, p, uniform, xlim, xlim)
# # # print M1
# # # print M3
# # # d1 = det(M1)
# # # d2 = det(M3)
# # # print (d2/d1)**(1/4)

# # # # x = linspace(-2, 2, 101)
# # # # pl.plot(x, sdvarcont(getf, p, xi, x))
# # # # pl.show()


# # nmax = 4*200
# # res = sequential(getf, p, xstart, xlim, nmax, optimize=False)
# # res = res[-(4*100):]
# # print res
# # pl.hist(res,20)
# # pl.show()

# # x = xi[0]
# # xl = linspace(xlim[0], xlim[1], 1001)
# # dxk = sdvarexact(getf, p, x, xl, False)
# # pl.plot(xl, dxk)
# # pl.show()

# # M3 = Mcontsub(getf, p, uniform, xlim, xlim)
# # print M1
# # print M3
# # d1 = det(M1)
# # d2 = det(M3)

# # print (d2/d1)**(1/3)
# xstart = array([ -3, -0.72, -0.72, 0, 0,  0.72, 0.72, 3])
# xlim = [-3, 3]
# x = linspace(xlim[0], xlim[1], 101)
# sd = sdvarexact(getf, p, xstart, x)
# pl.plot(x, sd)
# pl.show()

# # a = getf(p, x[0])
# # print a
# # print dot(a.T, a)



# lorentz = lambda p, x: p[2]/((x-p[0])**2 + p[1]**2) + p[3]
# # pl.plot(x, lorentz(p, x))
# # pl.plot(xi[0,:], lorentz(p, xi[0,:]),'o')
# # pl.show()


# # x = linspace(-2, 2, 101)
# # pl.plot(x, sdvarcont(getf, p, xi, x))
# # pl.show()


# # xi = [-a, 0, a, 0, -a, a]
# # x = linspace(-1, 1, 101)
# # pl.plot(x, sdvarexact(getf, p, xi, x))
# # pl.show()

# # xstart = array([-1, -0.5, 0.5, 1])
# # xlim = [-3, 3]
# # nmax = 100
# # res = sequential(getf, p, xstart, xlim, nmax, optimize=False)
# # res = res[-80:]
# # print res
# # pl.hist(res)
# # pl.show()

# # print det(Mcontinuous(getf, p, xi))
# # print det(Mexact(getf, p, x))

# # al = linspace(0.1, 2, 500)
# # ret = array([])
# # for a in al:
# #     # xi = array([[-a, 0, a], [1/3, 1/3, 1/3]])
# #     # xi = array([[-a, -a/2, 0, a/2, a], [1/10, 3/10, 2/10, 3/10, 1/10]])
# #     ret = append(ret, det(Mcontinuous(getf, p, xi)))

# # mm = argmax(ret)
# # print "%g => %g" %(al[mm], ret[mm])
# # pl.plot(al, ret)
# # pl.show()





# # ######
# # def uniform(x, pars):
# #     prob = 1 / (pars[0] - pars[1])
# #     return prob

# # def singleexp(x, pars):
# #     prob = pars[2]*exp(-(x-pars[0])**2/(2*pars[1]**2))/sqrt(2*pi*pars[1]**2)
# #     return prob

# # def compoundexp(x, pars):
# #     """
# #     e.g. pars = ((-a, b, 1), (a, b, 1))
# #     """
# #     h = 0
# #     for par in pars:
# #         try:
# #             prob += par[2]*exp(-(x-par[0])**2/(2*par[1]**2))/sqrt(2*pi*par[1])
# #         except NameError:
# #             prob = par[2]*exp(-(x-par[0])**2/(2*par[1]**2))/sqrt(2*pi*par[1])
# #         h += par[2]
# #     prob = prob /h
# #     return prob

# # xlim = [-4, 4]
# # # pars = [0, 0.1, 1]
# # # M2 = Mcontsub(getf, p, singleexp, xlim, pars)


# # b = 0.15
# # pars = [[-a, b, 1], [0, b, 1], [a, b, 1]]

# # a = 1.5
# # pars = [[-a, b, 1], [-a/2, b, 1], [a/2, b, 1], [a, b, 1]]

# # xi = array([[-a, -a/2, a/2, a], [1/4, 1/4, 1/4, 1/4]])
# # # M2 = Mcontsub(getf, p, compoundexp, xlim, pars)

# # M1 = Mcontinuous(getf, p, xi)
# # M3 = Mcontsub(getf, p, uniform, xlim, xlim)
# # print M1
# # print M3

# # # M1 = M1 + diag([1e-6]*4)
# # # print M1
# # print det(M1)
# # print det(M3)
# # # s1 = sqrt(linalg.eigh(inv(M1))[0])
# # # s3 = sqrt(linalg.eigh(inv(M3))[0])
# # # print s1
# # # print s3
# # # print [x/y for x, y in zip(s1, s3)]
# # # print prod(s1)
# # # print prod(s3)

# # # print det(M2)/det(M1)
# # print det(M3)/det(M1)
# # x = linspace(xlim[0], xlim[1], 101)
# # # pl.plot(x, compoundexp(x, pars))
# # # pl.show()


# ######### X0 robustness
# # xn = 21
# # x0p = linspace(0, 1, xn)
# # res = array([])
# # for x0 in x0p:
# #     z = x0
# #     a = 0.774
# #     xi = array([[-a+z, 0+z, a+z], [1/3, 1/3, 1/3]])
# #     MX = Mcontinuous(getf, p, xi)
# #     res = append(res, det(MX)/det(M3))

# # pl.plot(x0p, res)

# # x0p = linspace(0, 1, xn)
# # res = array([])
# # for x0 in x0p:
# #     z = x0
# #     a = 0.774
# #     b = 0.3
# #     pars = [[-a+z, b, 1], [0+z, b, 1], [a+z, b, 1]]
# #     MX = Mcontsub(getf, p, compoundexp, xlim, pars)
# #     res = append(res, det(MX)/det(M3))

# # pl.plot(x0p, res)

# # pl.plot([x0p[0], x0p[-1]], [1, 1])
# ##############

# # xn = 21
# # x0p = linspace(0.3, 2, xn)
# # res = array([])
# # for x0 in x0p:
# #     z = x0
# #     a = 0.774
# #     xi = array([[-a/x0, 0/x0, a/x0], [1/3, 1/3, 1/3]])
# #     MX = Mcontinuous(getf, p, xi)
# #     res = append(res, det(MX)/det(M3))

# # pl.plot(x0p, res)

# # res = array([])
# # for x0 in x0p:
# #     z = x0
# #     a = 0.774
# #     b = 0.3
# #     pars = [[-a/x0, b, 1], [0/x0, b, 1], [a/x0, b, 1]]
# #     MX = Mcontsub(getf, p, compoundexp, xlim, pars)
# #     res = append(res, det(MX)/det(M3))

# # pl.plot(x0p, res)

# # pl.plot([x0p[0], x0p[-1]], [1, 1])

# # pl.show()
