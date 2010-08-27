from numpy import *
import pylab as pl

lorentz = lambda p, x: p[2]/((x-p[0])**2 + p[1]**2) + p[3]

# x = linspace(-3, 3, 101)
# p = [0, 1, 1, 0]

xx1 = [-1, -0.5, 0.5, 1]
xx2 = [-0.77, 0, 0.77]

# x0l = linspace(0, 2, 101)
# for x in xx2:
#     pl.plot(x0l,  array([lorentz([x0, 1, 1, 0], x) for x0 in  x0l]), label="x = %g"%(x))
# pl.legend()
# pl.show()

# gl = linspace(0.5, 2, 101)
# for x in xx1:
#     pl.plot(gl,  array([lorentz([g, 1, 1, 0], x) for g in  gl]), label="x = %g"%(x))
# pl.legend()
# pl.show()
