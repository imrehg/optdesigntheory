from __future__ import division
from numpy import *
from scipy.odr import *
import pylab as pl

lorentz = lambda x, x0, g, A: A/((x-x0)**2+g**2)
lmodel = lambda beta, x: lorentz(x, beta[0], beta[1], beta[2])

x0 = 0
g = 1
A = 1

def x_uniform1(xlim, n):
    """ Uniform steps within the limits """
    return linspace(xlim[0], xlim[1], n)

def dofit(x, b0, sd):
    e = random.randn(1,len(x))*sd
    y1 = lorentz(x, b0[0], b0[1], b0[2]) + e[0]
    # y1 = lmodel(b0, x)
    # pl.plot(x, y1)
    # pl.show()
    data1 = Data(x, y1)
    model1 = Model(lmodel)
    odr1 = ODR(data1, model1, b0)
    odr1.set_job(fit_type=2)
    output1 = odr1.run()
    # output1.pprint()
    return output1

def doplot(x, b0, sd, title, dofig=False):
    x0_1 = []
    x0_1r = []
    g_1 = []
    g_1r = []
    a_1 = []
    a_1r = []
    for i in xrange(rep):
        out1 = dofit(x, b0, sd)
        x0_1.append(out1.beta[0])
        x0_1r.append(out1.sd_beta[0])
        g_1.append(out1.beta[1])
        g_1r.append(out1.sd_beta[1])
        a_1.append(out1.beta[2])
        a_1r.append(out1.sd_beta[2])

    if dofig:
        pl.figure()
        pl.subplot(2,2,1)
        pl.hist(x0_1)
        pl.title(title)
        pl.subplot(2,2,2)
        pl.hist(x0_1r)
        pl.title("Median value: %f" %median(x0_1r))
        pl.xlabel('Fitted x0 variance')
        pl.subplot(2,2,3)
        pl.hist(g_1)
        pl.subplot(2,2,4)
        pl.hist(g_1r)
        pl.title("Median value: %f" %median(g_1r))
        pl.xlabel('Fitted gamma variance')
    return [x0_1r, g_1r, a_1r]

xlim = [-4, 4]
n2 = 33
n = 330
rep = 200
b0 = [x0, g, A]
sd = 1/20

##############
# Compare uniform, 3-point, and my design
#########
x = x_uniform1(xlim, n)
r1 = doplot(x, b0, sd, 'Uniform, unique X', False)
print "Uniform:", median(r1[0]), median(r1[1]), median(r1[2])


### 6-point
# nx = 55
# xopt = [-1.58, -0.68, -0.2, 0.2, 0.68, 1.58]
# xopt = [0.68, -0.2,   0.2,   1.58, -1.58, -0.68]
### 3point
nx = 110
# X0 optimal
xopt = [-0.576, 0, 0.576]
xopt = [-0.78, 0, 0.78]
# G optimal
# xopt = [-1.18, 0, 1.18]
x = []
for xop in xopt:
    x = append(x, [xop]*nx)
x = array(x)
# x = array(append(append([-xopt]*nx, [0]*nx), [xopt]*nx))
r1 = doplot(x, b0, sd, 'Uniform, unique X', False)
print "3-point:", median(r1[0]), median(r1[1]), median(r1[2])

# X0 optimality
dlx0 = lambda x, x0, g: (2*(x-x0)/((x-x0)**2+g**2)**2)**2
x = []
while len(x) < n:
    xt = random.uniform(xlim[0]*g, xlim[1]*g)
    y = random.uniform(0, 3)
    if (y < dlx0(xt, 0, g*0.8)):
        x.append(xt)
x = sort(array(x))
r1 = doplot(x, b0, sd, 'Uniform, unique X', False)
print "X0 opt :", median(r1[0]), median(r1[1]), median(r1[2])

# Gamma optimality
dlg = lambda x, x0, g: (2*g/((x-x0)**2+g**2)**2)
x = []
while len(x) < n:
    xt = random.uniform(xlim[0]*g, xlim[1]*g)
    y = random.uniform(0, 3)
    if (y < dlg(xt, 0, 1)):
        x.append(xt)
x = sort(array(x))
r1 = doplot(x, b0, sd, 'Uniform, unique X', False)
print "G opt  :", median(r1[0]), median(r1[1]), median(r1[2])

# Amplitude optimality
dlg = lambda x, x0, g: (-2*g/((x-x0)**2+g**2)**2)**2
x = []
while len(x) < n:
    xt = random.uniform(xlim[0]*g, xlim[1]*g)
    y = random.uniform(0, 5)
    if (y < dlg(xt, 0, 1)):
        x.append(xt)
x = sort(array(x))
r1 = doplot(x, b0, sd, 'Uniform, unique X', False)
print "A opt  :", median(r1[0]), median(r1[1]), median(r1[2])

##############
# end comparison
#########


# dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)
# dlg = lambda x, x0, g: (((x-x0)**2-g**2)/((x-x0)**2+g**2)**2/pi)

# x = x_uniform1(xlim, n)

# # X0 optimality
# dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# x = []
# while len(x) < n:
#     xt = random.uniform(xlim[0]*g, xlim[1]*g)
#     y = random.uniform(0, 3)
#     if (y < dlx0(xt, 0, g*0.8)):
#         x.append(xt)
# x = sort(array(x))

# e = random.randn(1,len(x))*sd
# y1 = lorentz(x, b0[0], b0[1]) + e[0]

# x0l = linspace(-1, 1, 101)
# L = []
# for xo in x0l:
#     L.append(-0.5*sum((y1 - lorentz(x, xo, 1))**2))
# L = array(L)
# dL = diff(L)
# h = x0l[1] - x0l[0]
# xx = x0l[1:]-h/2

# # pl.plot(x0l, L)
# pl.plot(xx, dL**2)
# ddL = diff(dL)
# xxx = x0l[1:-1]
# pl.plot(xxx, -ddL)
# print 1/interp(xx[argmin(dL**2)], xxx, -ddL)
# pl.show()


# # ##### Gamma optimality
# # dlg = lambda x, x0, g: (((x-x0)**2-g**2)/((x-x0)**2+g**2)**2/pi)**2
# # x = []
# # while len(x) < n:
# #     xt = random.uniform(xlim[0]*g, xlim[1]*g)
# #     y = random.uniform(0, 3)
# #     if (y < dlg(xt, 0, 1)):
# #         x.append(xt)
# # x = sort(array(x))

# # e = random.randn(1,len(x))*sd
# # y1 = lorentz(x, b0[0], b0[1]) + e[0]

# # gl = linspace(0.9, 1.1, 101)
# # L = []
# # for g in gl:
# #     L.append(-0.5*sum((y1 - lorentz(x, 0, g))**2))
# # L = array(L)
# # dL = diff(L)
# # h = gl[1] - gl[0]
# # xx = gl[1:]-h/2

# # # pl.plot(x0l, L)
# # pl.plot(xx, dL**2)
# # ddL = diff(dL)
# # xxx = gl[1:-1]
# # pl.plot(xxx, -ddL)

# # print 1/interp(xx[argmin(dL**2)], xxx, -ddL)
# # pl.show()



# # gl = linspace(0.7, 1.3, 101)
# # L = []
# # for g in gl:
# #     L.append(-0.5*sum((y1 - lorentz(x, 0, g))**2))
# # L = array(L)
# # pl.plot(gl, L)
# # pl.show()


# # pl.plot(x, y1)
# # pl.show()

# # pl.plot(x, dlx0(x, x0, g)**2, label="(d/dx0)^2")
# # pl.plot(x, dlg(x, x0, g)**2, label="(d/dg)^")
# # pl.plot(x, abs(dlx0(x, x0, g)*dlg(x, x0, g)), label="d^2/(dgdx0)")
# # pl.legend(loc="best")

# # dd2 = lambda x, x0, g: abs(dlx0(x, x0, g)*dlg(x, x0, g))

# # ### ususal suspects: uniform along x
# # x = x_uniform1(xlim, n)
# # sd1 = sqrt(mean(dlx0(x, x0, g)**2))
# # print sd1

# # # x = x_uniform1(xlim, n)
# # # r1 = doplot(x, b0, sd, 'Uniform, unique X')
# # # r1m = median(r1[0])

# # # # ### Ususal suspects: uniform along x
# # # # r = int(n / n2)
# # # # xx = x_uniform1(xlim, n2)
# # # # x = []
# # # # for i in xrange(n2):
# # # #     for j in xrange(r):
# # # #         x.append(xx[i])
# # # # x = array(x)
# # # # r2 = doplot(x, b0, sd, 'Uniform, non-unique X')

# # # ### Non-uniform, unique X, density as dlx0
# # # dlx0 = lambda x, x0, g: abs(2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)
# # # x = []
# # # b0 = [0.1, 0.2]
# # # while len(x) < n:
# # #     xt = random.uniform(xlim[0], xlim[1])
# # #     y = random.uniform(0, 0.25)
# # #     if (y < dlx0(xt, x0, 1)):
# # #         x.append(xt)
# # # x = sort(array(x))
# # # r3 = doplot(x, b0, sd, 'Non-uniform, unique X, |dlx0|')
# # # pl.figure()
# # # pl.hist(x)


# # # # ### Non-uniform, unique X, density as dlx0^2
# # # dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# # # gl = linspace(0.1, 2, 11)
# # # x0 = 0
# # # impr = []
# # # for g in gl:
# # #     b0 = [0, 1]
# # #     x = []
# # #     while len(x) < n:
# # #         xt = random.uniform(xlim[0]*g, xlim[1]*g)
# # #         y = random.uniform(0, 3)
# # #         if (y < dlx0(xt, 0, g)):
# # #             x.append(xt)
# # #     x = sort(array(x))
# # #     r4 = doplot(x, b0, sd, 'Non-uniform, unique X, dlx0^2')
# # #     impr.append(median(r4[0]))
# # # pl.figure
# # # pl.plot(gl, array(impr)/r1m)


# # # dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# # # x0 = 0
# # # impr = []
# # # b0 = [0, 1]
# # # m = 2
# # # x = []
# # # while len(x) < n:
# # #     xt = random.uniform(xlim[0]*m, xlim[1]*m)
# # #     y = random.uniform(0, 3)
# # #     if (y < dlx0(xt, 0, 1*m)):
# # #         x.append(xt)
# # # x = sort(array(x))
# # # out1 = dofit(x, b0, sd)
# # # print out1.beta[0]
# # # print out1.sd_beta[0]
# # # # pl.figure
# # # # pl.plot(gl, impr)




# # # dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# # # x0l = linspace(0.1, 2, 21)
# # # g = 1
# # # impr = []
# # # for x0 in x0l:
# # #     x = []
# # #     while len(x) < n:
# # #         xt = random.uniform(xlim[0], xlim[1])
# # #         y = random.uniform(0, 3)
# # #         if (y < dlx0(xt, 0, 1)):
# # #             x.append(xt)
# # #     x = sort(array(x))
# # #     r4 = doplot(x, b0, sd, 'Non-uniform, unique X, dlx0^2')
# # #     impr.append(median(r4[0]))
# # # pl.figure
# # # pl.plot(gl, impr)

# # # pl.figure()
# # # pl.hist(x,bins=30)

# # # ### Non-uniform, unique X, density as dlx0^2
# # # x = []
# # # while len(x) < n:
# # #     xt = random.uniform(xlim[0], xlim[1])
# # #     y = random.uniform(0, 1)
# # #     if (y < dd2(xt, x0, g)):
# # #         x.append(xt)
# # # x = sort(array(x))
# # # r4 = doplot(x, b0, sd, 'Non-uniform, unique X, dd2')



# # # ### Non-uniform, unique X, density as dlg^2
# # # dlg = lambda x, x0, g: (((x-x0)**2-g**2)/((x-x0)**2+g**2)**2/pi)**2
# # # x = []
# # # while len(x) < n:
# # #     xt = random.uniform(xlim[0], xlim[1])
# # #     y = random.uniform(0, 0.25)
# # #     if (y < dlg(xt, x0, g)):
# # #         x.append(xt)
# # # x = sort(array(x))
# # # r5 = doplot(x, b0, sd, 'Non-uniform, unique X, dlg^2')

# # # ## Uniform X, density as dlx0^2
# # # x0l = linspace(0, 2, 31)
# # # g = 1
# # # impr = []
# # # dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# # # for x0 in x0l:
# # #     xx = linspace(xlim[0], xlim[1], n2)
# # #     p1 = dlx0(xx, 0, g)
# # #     p1 = p1 / sum(p1)
# # #     px = rint(p1 * n)
# # #     x = []
# # #     for i, p in enumerate(px):
# # #         for xn in xrange(int(p)):
# # #             x.append(xx[i])
# # #     x = array(x)
# # #     sd6 = sqrt(mean(dlx0(x, x0, 0.3)))
# # #     impr.append(sd1/sd6)
# # # pl.plot(x0l, impr)
# # # pl.ylabel('Relative variance')
# # # pl.ylabel("|x0' - x0|")

# # # dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# # # x0l = linspace(0, 2, 11)
# # # g = 1
# # # impr = []
# # # for x0 in x0l:
# # #     b0 = [x0, g]
# # #     x = []
# # #     while len(x) < n:
# # #         xt = random.uniform(xlim[0], xlim[1])
# # #         y = random.uniform(0, 1)
# # #         if (y < dlx0(xt, 0, g)):
# # #             x.append(xt)
# # #     x = sort(array(x))
# # #     r4 = doplot(x, b0, sd, 'Non-uniform, unique X, dlx0^2')
# # #     impr.append(median(r4[0]))
# # # pl.plot(x0l, impr)

# # # dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# # # gl = linspace(0.5, 2, 11)
# # # x0 = 0
# # # impr = []
# # # for g in gl:
# # #     b0 = [x0, g]
# # #     x = []
# # #     while len(x) < n:
# # #         xt = random.uniform(xlim[0], xlim[1])
# # #         y = random.uniform(0, 1)
# # #         if (y < dlx0(xt, x0, 1)):
# # #             x.append(xt)
# # #     x = sort(array(x))
# # #     r4 = doplot(x, b0, sd, 'Non-uniform, unique X, dlx0^2')
# # #     impr.append(median(r4[0]))
# # # pl.plot(gl, impr)


########### 3point design robustness test
# nx = 110
# # X0 optimal
# xopt = 0.576
# # G optimal
# #xopt = 1.18
# x = array(append(append([-xopt]*nx, [0]*nx), [xopt]*nx))

# x0l = linspace(0, 2, 31)
# impr = []
# mr1 = median(r1[0])
# for x0 in x0l:
#     b0 = [x0, g, A]
#     r2 = doplot(x, b0, sd, 'Uniform, unique X', False)
#     mr2 = median(r2[0])
#     print mr2
#     impr.append(mr2/mr1)
#     # print "X0 opt :", median(r1[0]), median(r1[1]), median(r1[2])
# #     xx = linspace(xlim[0]+x0, xlim[1]+x0, n2)
# #     p1 = dlx0(xx, x0, 1)
# #     p1 = p1 / sum(p1)
# #     px = rint(p1 * n)
# #     x = []
# #     for i, p in enumerate(px):
# #         for xn in xrange(int(p)):
# #             x.append(xx[i])
# #     x = array(x)
# #     sd6 = sqrt(mean(dlx0(x, 0, 1)))
# #     impr.append(sd1/sd6)
# pl.figure()
# pl.plot(x0l, impr)
# pl.ylabel('Relative variance')
# pl.xlabel("$\hat x_0 - x_0$")
# pl.ylim([0, 1.5])
# pl.savefig("rel_x0_3point.eps")
# pl.show()


# # x0 = 0
# # gl = linspace(0.2, 2, 101)
# # impr = []
# # dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# # for g in gl:
# #     xx = linspace(xlim[0]*g, xlim[1]*g, n2)
# #     p1 = dlx0(xx, 0, g)
# #     p1 = p1 / sum(p1)
# #     px = rint(p1 * n)
# #     x = []
# #     for i, p in enumerate(px):
# #         for xn in xrange(int(p)):
# #             x.append(xx[i])
# #     x = array(x)
# #     sd6 = sqrt(mean(dlx0(x, 0, 1)))
# #     impr.append(sd1/sd6)
# # pl.figure()
# # pl.plot(gl, impr)
# # pl.ylabel('Relative variance')
# # pl.xlabel("$\hat\Gamma / \Gamma$")
# # pl.savefig("rel_g.eps")
# # pl.show()


# # pl.show()
    
