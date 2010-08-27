from numpy import *
from scipy.linalg import inv, det
import pylab as pl


n1 = lambda p, x: p[0] + p[1]*exp(x) + p[2]*exp(-x)
n2 = lambda p, x: p[0] + p[1]*x + p[2]*x**2
# n1 = lambda p, x: p[1]*exp(x) + p[2]*exp(-x)
# n2 = lambda p, x: p[0]*x + p[1]*x**2

xc = linspace(-1, 1, 101)



dl11 = lambda x: x*0 + 1
dl12 = lambda x: exp(x)
dl13 = lambda x: exp(-x)

dl21 = lambda x: x*0 + 1
dl22 = lambda x: x
dl23 = lambda x: x**2



def getf1(x):
    return array([[dl11(x)], [dl12(x)], [dl13(x)]])
    # return array([[dl12(x)], [dl13(x)]])

def getf2(x):
    return array([[dl21(x)], [dl22(x)], [dl23(x)]])
    # return array([[dl22(x)], [dl23(x)]])



xa = array([[-1, -0.669, 0.144, 0.987], 
            [0.253, 0.428, 0.247, 0.072]])

#### Include the distribution
def getest(f1, f2, xa):
    x = xa[0,:]
    for i, xi in enumerate(x):
        try:
            M1 += dot(f2(xi), f1(xi).T)*xa[1, i]
        except:
            M1 = dot(f2(xi), f1(xi).T)*xa[1, i]
        try:
            M2 += dot(f2(xi), f2(xi).T)*xa[1, i]
        except:
            M2 = dot(f2(xi), f2(xi).T)*xa[1, i]
    return dot(inv(M2), M1)

def geteste(f1, f2, xa):
    x = xa
    N = len(xa)
    for i, xi in enumerate(x):
        try:
            M1 += dot(f2(xi), f1(xi).T)/N
        except:
            M1 = dot(f2(xi), f1(xi).T)/N
        try:
            M2 += dot(f2(xi), f2(xi).T)/N
        except:
            M2 = dot(f2(xi), f2(xi).T)/N
    return dot(inv(M2), M1)

def iterative(f1, ff1, f2, ff2, p1, xi, maxiter):

    x = xi
    xl = linspace(-1, 1, 201)
    for i in range(maxiter-len(xi)):
        FF = geteste(ff1, ff2, x)
        p2 = dot(FF, array(p1))
        psi = [(f1(p1, xv) - f2(p2, xv))**2 for xv in xl]
        m = argmax(psi)
        x = append(x, xl[m])
        print "%d : %f (%f)" %(i, xl[m], psi[m])
    return x

x = xa[0,:]
FF =  getest(getf1, getf2, xa)
p1 = [4.5, -0.5, -2]
p2 = dot(FF, array(p1))
# p2 = dot(FF, array(p1[1:]))
print p2

# p2[0] += 0
# pl.plot(xc, n1(p1, xc))
# pl.plot(xc, n2(p2, xc))
# pl.show()
xin = array([-1, -0.25, 0.25, 1])
xout = iterative(n1, getf1, n2, getf2, p1, xin, 500)
# pl.hist(xout)
# pl.show()

delta = 0
for i, xi in enumerate(xout):
    delta += (n1(p1, xi) - n2(p2, xi))**2
delta /= len(xout)
print delta

FF =  geteste(getf1, getf2, xout)
p2 = dot(FF, array(p1))

# delta = 0
# for i, xi in enumerate(x):
#     delta += (n1(p1, xi) - n2(p2, xi))**2 * xa[1, i]

# print delta

xl = linspace(-1, 1, 201)
psi = [(n1(p1, xi) - n2(p2, xi))**2 for xi in xl]
pl.plot(xl, psi)
pl.plot([xl[0], xl[-1]], [delta, delta], '--')
pl.show()
