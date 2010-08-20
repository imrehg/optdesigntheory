from __future__ import  division
from numpy import *
import pylab as pl

xout = loadtxt("xout.txt")
f = array(range(0, len(xout)))/len(xout)

# pl.plot(xout, f)
# pl.show()

xl = linspace(-2, 2, 31)
h = xl[1]-xl[0]
n = len(xout)
pdf = []
for i, x in enumerate(xl[:-1]):
    pdf.append(len(xout[(xout >= xl[i]) & (xout <= xl[i+1])]))

pl.plot(xl[:-1]+h/2, pdf)
pl.show()
