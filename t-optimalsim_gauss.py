from __future__ import division
from numpy import *
from scipy.odr import *
import pylab as pl
import sys


n1 = lambda p, x:  p[2]*p[1]/((x-p[0])**2 + p[1]**2)
n2 = lambda p, x:  p[2]/p[1]*exp(-(x-p[0])**2/(2*p[1]**2))

p = [0, 1, 1]

# pl.plot(x, n1(p, x), '.')
# pl.plot(x, n2(p, x), '.')
# pl.show()

# y = n1(p, x)
# data = Data(x, y)
# model = Model(n2)
# fit = ODR(data, model, p)
# fit.set_job(fit_type=2)
# output = fit.run()
# beta = output.beta

# pl.plot(x, n1(p, x), '.-')
# pl.plot(x, n2(beta, x), 'x-')
# pl.plot(x, (n1(p, x) - n2(beta, x))**2, 'x-')

rss = lambda f1, f2, p1, p2, x: (f1(p1, x) - f2(p2, x))**2
d2 = lambda f1, f2, p1, p2, x, w: sum(rss(f1, f2, p1, p2, x)*w)


def getfit(f, x, y, p):
    data = Data(x, y)
    model = Model(f)
    fit = ODR(data, model, p)
    fit.set_job(fit_type=2)
    output = fit.run()
    return output.beta


def iterative(f1, f2, p1, x0, maxiter, xlim, np):
    x = x0
    xt = linspace(xlim[0], xlim[1], np)

    for i in xrange(maxiter - len(x)):
        y = f1(p, x)
        beta = getfit(f2, x, y, p1)
        rssx = rss(f1, f2, p1, beta, xt)
        x = append(x, xt[argmax(rssx)])
    return x


p = [0, 1, 1]
x0 = array([-2, -1, 1, 2])
xlim = array([-3, 3])
np = 8001
maxiter = 10000

if len(sys.argv) > 1:
    # Generate data on fast computer
    xout = iterative(n1, n2, p, x0, maxiter, xlim, np)
    savetxt("xout.txt", xout)

else:

    # Analyse data on slow computer
    xout = loadtxt("xout.txt")
    xout = xout[2000:]
    maxiter = len(xout)
    pl.figure(1)
    nn, bin, patch = pl.hist(xout, 21)

    for i, n in enumerate(nn):
        print "%f -> %d (%f)" %(bin[i], n, n/maxiter)

    xc = linspace(xlim[0], xlim[1], 301)

    y = n1(p, xout)
    beta = getfit(n2, xout, y, p)
    print beta
    rssx = rss(n1, n2, p, beta, xc)
    pl.figure(2)
    pl.plot(xc, rssx)
    pl.plot(xout, rss(n1, n2, p, beta, xout), 'x')

    pl.figure(3)
    pl.plot(xout,'.')
    pl.show()

    #### Uniform
    # xc = linspace(xlim[0], xlim[1], 3001)
    # pl.plot(xc, n1(p, xc), '.')
    # beta = getfit(n2, xc, n1(p, xc), p)
    # pl.plot(xc, n2(beta, xc), '.')

    # print sum((n1(p, xc) - n2(beta, xc))**2) / len(xc)

    

