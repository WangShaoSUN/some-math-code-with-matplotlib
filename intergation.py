from __future__ import division  # Python 2 compatibility

def simpson(f, a, b, n):
    """Approximates the definite integral of f from a to b by the
    composite Simpson's rule, using n subintervals (with n even)"""

    if n % 2:
        raise ValueError("n must be even (received n=%d)" % n)

    h = (b - a) / n
    s = f(a) + f(b)

    for i in range(1, n, 2):
        s += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        s += 2 * f(a + i * h)

    return s * h / 3

# Demonstrate that the method is exact for polynomials up to 3rd order
print(simpson(lambda x:x**3, 0.0, 10.0, 2))       # 2500.0
print(simpson(lambda x:x**3, 0.0, 10.0, 100000))  # 2500.0

print(simpson(lambda x:x**4, 0.0, 10.0, 2))       # 20833.3333333
print(simpson(lambda x:x**4, 0.0, 10.0, 100000))  # 20000.0



def simpsons_rule(f,a,b):
    c = (a+b) / 2.0
    h3 = abs(b-a) / 6.0
    return h3*(f(a) + 4.0*f(c) + f(b))

def recursive_asr(f,a,b,eps,whole):
    "Recursive implementation of adaptive Simpson's rule."
    c = (a+b) / 2.0
    left = simpsons_rule(f,a,c)
    right = simpsons_rule(f,c,b)
    if abs(left + right - whole) <= 15*eps:
        return left + right + (left + right - whole)/15.0
    return recursive_asr(f,a,c,eps/2.0,left) + recursive_asr(f,c,b,eps/2.0,right)

def adaptive_simpsons_rule(f,a,b,eps):
    "Calculate integral of f from a to b with max error of eps."
    return recursive_asr(f,a,b,eps,simpsons_rule(f,a,b))

from math import sin
print adaptive_simpsons_rule(sin,0,1,.000000001)