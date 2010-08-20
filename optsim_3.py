from __future__ import division
from numpy import *
from scipy.odr import *
import pylab as pl

lorentz = lambda x, x0, g: g/((x-x0)**2+g**2)/pi
lmodel = lambda beta, x: lorentz(x, beta[0], beta[1])

x0 = 0
g = 1

def x_uniform1(xlim, n):
    """ Uniform steps within the limits """
    return linspace(xlim[0], xlim[1], n)

def dofit(x, b0, sd):
    e = random.randn(1,len(x))*sd
    y1 = lorentz(x, b0[0], b0[1]) + e[0]
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

def doplot(x, b0, sd, title):
    x0_1 = []
    x0_1r = []
    g_1 = []
    g_1r = []
    for i in xrange(rep):
        out1 = dofit(x, b0, sd)
        x0_1.append(out1.beta[0])
        x0_1r.append(out1.sd_beta[0])
        g_1.append(out1.beta[1])
        g_1r.append(out1.sd_beta[1])

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
    return [x0_1r, g_1r]

xlim = [-4, 4]
n2 = 33
n = 330
rep = 4000
b0 = [x0, g]
sd = 1/20

dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
### ususal suspects: uniform along x
x = x_uniform1(xlim, n)
sd1 = sqrt(mean(dlx0(x, x0, g)))
print sd1


# ### Ususal suspects: uniform along x
# r = int(n / n2)
# xx = x_uniform1(xlim, n2)
# x = []
# for i in xrange(n2):
#     for j in xrange(r):
#         x.append(xx[i])
# x = array(x)
# r2 = doplot(x, b0, sd, 'Uniform, non-unique X')

### Non-uniform, unique X, density as dlx0
# dlx0 = lambda x, x0, g: abs(2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)
# x = []
# b0 = [0.1, 0.2]
# while len(x) < n:
#     xt = random.uniform(xlim[0], xlim[1])
#     y = random.uniform(0, 0.25)
#     if (y < dlx0(xt, x0, 1)):
#         x.append(xt)
# x = sort(array(x))
# r3 = doplot(x, b0, sd, 'Non-uniform, unique X, |dlx0|')
# pl.figure()
# pl.hist(x)

# ### Non-uniform, unique X, density as dlx0^2
# dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# x = []
# while len(x) < n:
#     xt = random.uniform(xlim[0], xlim[1])
#     y = random.uniform(0, 1)
#     if (y < dlx0(xt, x0, 0.5)):
#         x.append(xt)
# x = sort(array(x))
# r4 = doplot(x, b0, sd, 'Non-uniform, unique X, dlx0^2')

# ### Non-uniform, unique X, density as dlg^2
# dlg = lambda x, x0, g: (((x-x0)**2-g**2)/((x-x0)**2+g**2)**2/pi)**2
# x = []
# while len(x) < n:
#     xt = random.uniform(xlim[0], xlim[1])
#     y = random.uniform(0, 0.25)
#     if (y < dlg(xt, x0, g)):
#         x.append(xt)
# x = sort(array(x))
# r5 = doplot(x, b0, sd, 'Non-uniform, unique X, dlg^2')

# ## Uniform X, density as dlx0^2
# x0l = linspace(0, 2, 31)
# g = 1
# impr = []
# dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# for x0 in x0l:
#     xx = linspace(xlim[0], xlim[1], n2)
#     p1 = dlx0(xx, 0, g)
#     p1 = p1 / sum(p1)
#     px = rint(p1 * n)
#     x = []
#     for i, p in enumerate(px):
#         for xn in xrange(int(p)):
#             x.append(xx[i])
#     x = array(x)
#     sd6 = sqrt(mean(dlx0(x, x0, 0.3)))
#     impr.append(sd1/sd6)
# pl.plot(x0l, impr)
# pl.ylabel('Relative variance')
# pl.ylabel("|x0' - x0|")

# dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# x0l = linspace(0, 2, 11)
# g = 1
# impr = []
# for x0 in x0l:
#     b0 = [x0, g]
#     x = []
#     while len(x) < n:
#         xt = random.uniform(xlim[0], xlim[1])
#         y = random.uniform(0, 1)
#         if (y < dlx0(xt, 0, g)):
#             x.append(xt)
#     x = sort(array(x))
#     r4 = doplot(x, b0, sd, 'Non-uniform, unique X, dlx0^2')
#     impr.append(median(r4[0]))
# pl.plot(x0l, impr)

# dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# gl = linspace(0.5, 2, 11)
# x0 = 0
# impr = []
# for g in gl:
#     b0 = [x0, g]
#     x = []
#     while len(x) < n:
#         xt = random.uniform(xlim[0], xlim[1])
#         y = random.uniform(0, 1)
#         if (y < dlx0(xt, x0, 1)):
#             x.append(xt)
#     x = sort(array(x))
#     r4 = doplot(x, b0, sd, 'Non-uniform, unique X, dlx0^2')
#     impr.append(median(r4[0]))
# pl.plot(gl, impr)




# x0 = 0
# gl = linspace(0.01, 2, 201)
# impr = []
# dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# for g in gl:
#     xx = linspace(xlim[0], xlim[1], n2)
#     p1 = dlx0(xx, x0, 1)
#     p1 = p1 / sum(p1)
#     px = rint(p1 * n)
#     x = []
#     for i, p in enumerate(px):
#         for xn in xrange(int(p)):
#             x.append(xx[i])
#     x = array(x)
#     sd6 = sqrt(mean(dlx0(x, x0, g)))
#     impr.append(sd1/sd6)
# pl.plot(gl, impr)
# pl.ylabel('Relative variance')
# pl.ylabel("|x0' - x0|")

pl.show()


# pl.show()
    
