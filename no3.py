from numpy import *
import pylab as pl

x = linspace(-3, 3, 101)
x0 = 0
g = 1
# dlx0 = lambda x, x0, g: (2*(x-x0)*g/((x-x0)**2+g**2)**2/pi)**2
# df = lambda x, x0, g: (x-x0)**2-g**2
# pl.plot(x, dlx0(x, 0, 1))
# # ddlx0 = diff(dlx0(x, x0, g))
# # dd = lambda x, x0, g: (-2*g*((x-x0)**2+g**2)+4*(x-x0)*g*(x-x0))/((x-x0)**2+g**2)**2/pi
# # h = x[1]-x[0]
# # xx = x[1:]-h/2
# # pl.plot(xx, ddlx0/h)
# # pl.plot(x, dd(x, x0, g))
# # pl.plot(x, df(x, x0, g))
# pl.show()

dlg = lambda x, x0, g: (((x-x0)**2-g**2)/((x-x0)**2+g**2)**2/pi)**2
pl.plot(x, dlg(x, x0, g))
pl.show()
