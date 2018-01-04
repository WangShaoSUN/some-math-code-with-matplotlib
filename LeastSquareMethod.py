#encoding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import pylab as pl
def func(p,x):
    k,b=p
    return k*x+b
def error(p,x,y):
    return  func(p,x)-y
X=np.arange(0,5,0.1)
Z=[3+5*x for x in X]
Y=[np.random.normal(z,0.5) for z in Z]


p0=[1,2]
Para=leastsq(error,p0,args=(X,Y))

k,b=Para[0]
print k,b
plt.title("least Square Method")
plt.plot(X,Y,"ro",label=u"sample")
x=np.arange(0,5,0.1)
y=k*x+b
plt.plot(x,y,color='black',label=u"fit curve",linewidth=3)

plt.legend(loc='lower right')
plt.savefig("lsm.jpg")
plt.show()


# def real_fun(x):
#     return np.sin(2*np.pi*x)
# def poly_fun(p,x):
#     f= np.poly1d(p)
#     return f(x)

