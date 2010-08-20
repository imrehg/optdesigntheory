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

### Ususal suspects: uniform along x
x = x_uniform1(xlim, n)
r1 = doplot(x, b0, sd, 'Uniform, unique X')

### Ususal suspects: uniform along x
r = int(n / n2)
xx = x_uniform1(xlim, n2)
x = []
for i in xrange(n2):
    for j in xrange(r):
        x.append(xx[i])
x = array(x)
r2 = doplot(x, b0, sd, 'Uniform, non-unique X')

### Non-uniform, unique X, density as dlx0
dlx0 = lambda x, x0, g: abs(2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)
x = []
while len(x) < n:
    xt = random.uniform(xlim[0], xlim[1])
    y = random.uniform(0, 0.25)
    if (y < dlx0(xt, x0, g)):
        x.append(xt)
x = sort(array(x))
r3 = doplot(x, b0, sd, 'Non-uniform, unique X, |dlx0|')

### Non-uniform, unique X, density as dlx0^2
dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
x = []
while len(x) < n:
    xt = random.uniform(xlim[0], xlim[1])
    y = random.uniform(0, 0.25)
    if (y < dlx0(xt, x0, g)):
        x.append(xt)
x = sort(array(x))
r4 = doplot(x, b0, sd, 'Non-uniform, unique X, dlx0^2')

### Non-uniform, unique X, density as dlg^2
dlg = lambda x, x0, g: (((x-x0)**2-g**2)/((x-x0)**2+g**2)**2/pi)**2
x = []
while len(x) < n:
    xt = random.uniform(xlim[0], xlim[1])
    y = random.uniform(0, 0.25)
    if (y < dlg(xt, x0, g)):
        x.append(xt)
x = sort(array(x))
r5 = doplot(x, b0, sd, 'Non-uniform, unique X, dlg^2')

## Uniform X, density as dlx0^2
n2 = 33
dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
xx = linspace(xlim[0], xlim[1], n2)
p1 = dlx0(xx, x0, g)
p1 = p1 / sum(p1)
px = rint(p1 * n)
x = []
for i, p in enumerate(px):
    for xn in xrange(int(p)):
        x.append(xx[i])
x = array(x)
r6 = doplot(x, b0, sd, 'Uniform, non-unique X, dlx0^2')


pl.figure()
pl.subplot(3, 2, 1)
pl.hist(r1[0])
pl.xlabel("(1) x0 estimate variance")
# pl.title("Uniform X, unique values")
pl.subplot(3, 2, 2)
pl.hist(r2[0])
pl.xlabel("(2) x0 estimate variance")
# pl.title("Uniform X, repeated values")

pl.subplot(3, 2, 3)
pl.hist(r3[0])
pl.xlabel("(3) x0 estimate variance")
# pl.title("Non-uniform X, density as d/dx0")
pl.subplot(3, 2, 4)
pl.hist(r4[0])
pl.xlabel("(4) x0 estimate variance")
# pl.title("Non-uniform X, density as (d/dx0)^2")

pl.subplot(3, 2, 5)
pl.hist(r5[0])
pl.xlabel("(5) x0 estimate variance")
# pl.title("Non-uniform X, density as d/dg")
pl.subplot(3, 2, 6)
pl.hist(r6[0])
pl.xlabel("(6) x0 estimate variance")
# pl.title("Uniform X, repeated density as (d/dx0)^2")

pl.show()
    
