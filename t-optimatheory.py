from numpy import *
from scipy.stats import chi2
import pylab as pl


# delta = 7.3e-3
delta = 4e-3
s = 0.25
c = 0.05

N = array(range(5,2000))
x1 = N*delta/s**2 + N
x2 = N
c1 = 1-chi2.cdf(x1[30:], N[30:]-3-1)
m = argmin(abs(c1-c))
Ncrit = N[m]
print Ncrit
# pl.plot(N, x)
pl.plot(N[30:], c1)
pl.plot([N[0], N[-1]], [0.05, 0.05], '--')
# pl.plot(N, c2)
pl.show()
