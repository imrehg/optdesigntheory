from numpy import *
import pylab as pl
from scipy.odr import *


def dofit(f, x, y, p0):
    data = Data(x, y)
    model = Model(f)
    fit = ODR(data, model, p0)
    fit.set_job(fit_type=2)
    output = fit.run()
    return output


lorentz = lambda p, x: p[2] * p[1] / ((x-p[0])**2 + p[1]**2)

p = [0, 1, 1]

nrep = 100000

xc = linspace(-3, 3, 201)
sd = 0.1
x0s = array([])
gs = array([])
for nr in xrange(nrep):
    y = lorentz(p, xc) + random.randn(len(xc))*sd
    out = dofit(lorentz, xc, y, p)
    x0s = append(x0s, out.sd_beta[0])
    gs = append(gs, out.sd_beta[1])

xi = [-0.775, 0, 0.775]
xd = array([])
for x in xi:
    xd = append(xd, [x]*67)
x0sd = array([])
gsd = array([])
for nr in xrange(nrep):
    y = lorentz(p, xd) + random.randn(len(xd))*sd
    out = dofit(lorentz, xd, y, p)
    x0sd = append(x0sd, out.sd_beta[0])
    gsd = append(gsd, out.sd_beta[1])


xi = [-0.576, 0.576]
xds = array([])
for x in xi:
    xds = append(xds, [x]*99)
xds = append(xds, [0]*3)
x0sds = array([])
gsds = array([])
for nr in xrange(nrep):
    y = lorentz(p, xds) + random.randn(len(xds))*sd
    out = dofit(lorentz, xds, y, p)
    x0sds = append(x0sds, out.sd_beta[0])
    gsds = append(gsds, out.sd_beta[1])
print max(x0sds)
h = linspace(0, 0.03, 91)
pl.figure(1)
nn1, bin1, patch1 = pl.hist(x0s, h)
nn2, bin2, patch2 = pl.hist(x0sd, h)
nn3, bin3, patch3 = pl.hist(x0sds, h)

out = zip(bin1, nn1, nn2, nn3)
savetxt('histogram.txt', out)
# pl.figure(2)
# pl.hist(gs)
# pl.hist(gsd)
# pl.hist(gsds)
pl.savefig('simdone.eps')
# pl.show()
