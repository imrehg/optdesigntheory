from numpy import *
import pylab as pl


gauss = lambda x, p: exp(-(x-p[0])**2/(2*p[1]**2))/sqrt(2*pi*p[1])

# x = linspace(-3, 3, 100)
# p = [0, 1]
# h = x[1]-x[0]
# dx = diff(gauss(x, p))/h

# hh = h*2
# ddx = diff(dx)/hh

# # pl.plot(x, gauss(x, p), label='original')
# pl.plot(x[1:]-h/2, abs(dx), ':', label='1st')
# pl.plot(x, abs(dgauss(x, p)))
# #pl.plot(x[2:]-h, abs(ddx), '--', label='2nd')
# pl.legend(loc='best')
# pl.show()


# x0 = linspace(-4, 4, 201)
# s = 1
# x = 1

# y = array([gauss(x, [u, s]) for u in x0])
# h = x0[1]-x0[0]
# dy = (diff(y)/h)**2

# dyt = ((x-x0)/s**2*y)**2

# pl.title('Variance of centre')

# # pl.plot(x0, y)
# pl.plot(x0[1:]-h/2, dy)
# #pl.plot(x0, dyt)
# pl.show()


lorentz = lambda x, x0, g: g/((x-x0)**2+g**2)/pi
# g = linspace(1,5,31)
# x0 = 0
# x = 3
# y = lorentz(x, x0, g)
# h = g[1]-g[0]
# dy = diff(y)/h

# Good: dLorentz/dx0
dlx0 = lambda x, x0, g: 2*(x-x0)*g/((x-x0)**2+g**2)**2/pi
# Good: dLorentz/dg
dlg = lambda x, x0, g: ((x-x0)**2-g**2)/((x-x0)**2+g**2)**2/pi


# pl.plot(g[1:]-h/2, dy)
# pl.plot(g, dlg(x, x0, g))


# x0l = linspace(-4, 4, 3)
# x = linspace(-10, 10, 1001)
# g = 1
# for x0 in x0l:
#     pl.plot(x, dlx0(x, x0, g)**2, label="%d"%x0)

# gl = linspace(1, 12, 10)
# x = linspace(-10, 10, 1001)
# x0 = 0
# for g in gl:
#     pl.plot(x, dlg(x, x0, g)**2*g**4, label="%d"%g)

x0 = 0
g = 1
x = linspace(-3.4, 3.4, 1001)
px0 = dlx0(x, x0, g)
pg = dlg(x, x0, g)
together = ((px0**2+pg**2)/2)
px0 /= max(abs(px0))
pg /= max(abs(pg))
together /= max(together)
pl.plot(x, ((px0)**2), '-', linewidth=4, label='Centre')
pl.plot(x, ((pg)**2), '--', linewidth=4, label='Width')
pl.plot(x, sqrt(together), '-', linewidth=5, label='Centre & Width')
pl.plot(x, lorentz(x, x0, g)/lorentz(x0, x0, g))
pl.xlabel('x')
pl.ylabel('Relative information')
pl.ylim([0, 1.1])
pl.xlim([x[0], x[-1]])


# pl.plot(x, lorentz(x, x0, g)/lorentz(x0, x0, g))

pl.legend(loc='best')
pl.show()

