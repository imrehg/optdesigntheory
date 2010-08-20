from numpy import *
from scipy.optimize import fmin
import pylab as pl

dlx0 = lambda x, x0, g: (2*(x-x0)/((x-x0)**2+g**2)**2)**2
dlg = lambda x, x0, g: (2*g/((x-x0)**2+g**2)**2)


ds = 1
sd2 = 0.3
n = 50
xstart = linspace(-2, 2, n)
x0 = 0
g = 1


# opt1 = lambda x: -sqrt(mean(dlx0(x, x0, g)**2))
# print opt1(xstart)
# xopt = fmin(opt1, xstart, xtol=1e-8, maxfun=40000)
# print xopt
# print opt1(xopt)
# pl.hist(xopt)
# pl.show()

nrep = 500


def opt2(x):
    A = (mean(dlx0(x, x0, g)))**2
    B = 2*sd2*mean(dlx0(x, x0, g)*dlg(x,x0,g))
    C = (mean(dlg(x, x0, g))*sd2)**2 - ds**2
    return (-B + sqrt(B**2 - 4*A*C))/(2*A)



# print opt2(xstart)
xout = array([])
for i in xrange(nrep):
    xopt = fmin(opt2, xstart, xtol=1e-8, maxfun=2000000)
    xout = append(xout, xopt)
# print xopt
# print opt2(xopt)
# pl.hist(xopt)
# xopt = sort(xopt)
xout = sort(xout)
# savetxt("xout.txt", xout)
pl.plot(xout, 'x')
#pl.hist(xout, 30)
pl.show()

