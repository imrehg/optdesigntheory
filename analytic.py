from __future__ import division
import numpy as np
import pylab as pl
from sympy import *

a = Symbol("a")
d1 = 2*a/(a**2 + 1)**2
d2 = (a**2 - 1) / (a**2 + 1)**2
d3 = 1/(a**2 + 1)
x = Matrix([[d1, d2, d3]])
x1 = x.T * x
x2 = x1.subs(a, 0)
x3 = x1.subs(a, -a)
xx = (x1 + x2 + x3)/3
# print xx
dxx = xx.det()

# ddet = diff(dxx, a)
# alist = np.linspace(0.7745, 0.775, 1001)
# res = np.array([])
# for atest in alist:
#     res = np.append(res, ddet.subs(a, atest))
# mmin = np.argmin(abs(res))
# print "%f" %alist[mmin]
# pl.plot(alist, res)
# pl.show()
print dxx.subs(a, 0.7746)


# print Dot(x.T, x)
# b = Matrix([[8*a*a/(3*(a**2 + 1)**4), 0, 0],
#             [0, 2*(a**2-1)**2/3/(a**2+1)**4+1/3, 2*(a**2-1)/3/(a**2+1)**3+1/3],
#             [0, 2*(a**2-1)/3/(a**2+1)**3+1/3, 2/3/(a**2+1)**2+1/3]])
# db = b.det()
# ddet = diff(db, a)
# print ddet
