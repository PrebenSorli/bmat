# LaGrange interpolation
# inputs: n nodes: x_i and n values: y_j, interval bounds: a,b
import math
from itertools import product
import autograd.numpy as np
from autograd import grad
from numpy import linalg as LA
import matplotlib.pyplot as plt
from functools import partial


def LaGrange(X, Y):
    n = len(X)
    L = np.poly1d([0])
    for i in range(n):
        p = np.poly1d(Y[i])
        for j in range(n):
            if i != j:
                factor = X[i] - X[j]
                p = np.polymul(p, np.poly1d([1, -X[j]]) / factor)
        L = np.polyadd(L, p)
    return L


# A is the x values for which we want to plot.
def interpolate(X, Y, start, end, iters):
    L = LaGrange(X, Y)
    A = equidist(start, end, iters)
    return [L(A[i]) for i in range(iters)]


def LaGrange2(X, f, x):
    n = len(X)
    L = []
    terms = []
    np.array(L)
    np.array(terms)
    for k in range(n):
        factors = [(x - X[i]) / (X[k] - X[i]) for i in range(n) if (X[i] != X[k])]
        L.append(np.product(factors))
        terms.append(L[k]*f(X[k]))
    return lambda x: sum(terms)


def eval(X, f, x):
    return LaGrange2(X, f, x)(x)


# A is the x values for which we want to plot.
def interpolate2(X, f, start, end, iters):
    A = equidist(start, end, iters)
    return [eval(X, f, A[i]) for i in range(iters)], A


def piecewiseLaGrange(a, b, K, x, n, f):
    i = a
    X = []
    while i < b:
        X.append([i, i + (b - a) / K])
        i += (b - a) / K
    for y in X:
        if x < y[1]:
            return eval(equidist(y[0], y[1], n), f, x)


# A is the x values for which we want to plot.
def interpolate_piecewise(start, end, iters, K, n, f):
    A = equidist(start, end, iters)
    return [piecewiseLaGrange(start, end, K, A[i], n, f) for i in range(iters)]


# a,b interval bounds, n number of nodes
def equidist(a, b, n):
    return np.asarray(np.linspace(a, b, n))


def chebyshev(a, b, n):
    return np.asarray([1 / 2 * (a + b) + 1 / 2 * (b - a) * math.cos((2 * k - 1) / (2 * n) * math.pi) for k in range(1, n+1)])


def runge(x):
    return 1/(x**2+1)


def rad_function(x):
    return 3/4 * (np.exp(-((9 * x - 2)**2) / 4) + np.exp(-((9*x-2)**2)/49)) + 1/2 * (np.exp((-(9*x-7)**2)/4)) - 1/10*(np.exp((9*x-4)**2))


def approxInfNorm(fx, p_n):
    return np.linalg.norm(abs(fx - p_n), np.inf)


def approx2norm(fx, p_n):
    return np.linalg.norm(abs(fx - p_n), 2)


def func_values(f, X):
    return np.asarray([f(x) for x in X])


def func_values2(f, X):
    return np.asarray([[x,f(x)] for x in X])


def F(x):
    return np.cos(2 * np.pi * x)


def G(x):
    return np.exp(3 * x) * np.sin(2 * x)


def error_vs_n(N, f, norm):
    M = []
    theoretic = []
    for i in range(1, N):
        X = equidist(0, 1, 100 * i)
        fx = func_values(f, X)
        p_n,_ = interpolate2(chebyshev(0, 1, i), f, 0, 1, 100 * i)
        M.append(norm(fx, p_n))
        theoretic.append((2 * np.pi)**(i + 1) / (math.factorial(i + 1)))
    return M, theoretic


def expError(N, f, norm):
    M = []
    for i in range(1, N):
        X = equidist(0, np.pi/4, 100 * i)
        fx = func_values(f, X)
        p_n,_ = interpolate2(chebyshev(0, np.pi/4, i), f, 0, np.pi/4, 100 * i)
        M.append(norm(fx, p_n))
    return M


def expError2(K, norm, n):
    M = []
    X = equidist(0, math.pi / 4, 100 * n)
    for i in range(1, K):
        fx = func_values(G, X)
        p_n = interpolate_piecewise(0, math.pi / 4, 100 * n, i, n, G)
        M.append(norm(fx, p_n))
    return M


def p():
    """
    f er funksjon
    a er start
    b er slutt
    N er antall kjente noder
    K er de kjente nodene med elementer på formen (x)
    """
    p.f, p.a, p.b, p.N, p.K = F, 0., 1., 1000, nodes2


def cost(X):
    cost = 0
    for k in range(0, len(p.K)):
        cost = cost + (p.f(k)-eval(X, p.f, p.K[k]))**2
    return (p.b-p.a)/p.N*cost


def p2():
    """
      f er funksjon
      a er start
      b er slutt
      N er antall kjente noder
      K er de kjente nodene med elementer på formen (x)
      """
    p2.f, p2.a, p2.b, p2.N, p2.K = runge, 0., 1., 1000, nodes2


def phi(r, epsilon):
    return np.exp(-(epsilon*r)**2)


def weight(X):
    M = np.array([[phi(abs(X[i] - X[j]), X[len(X) - 1]) for j in range(len(X) - 1)] for i in range(len(X) - 1)])
    W = np.linalg.solve(M, func_values(p2.f, nodes))
    return W


def radial_basis(X, e):
    sum = 0
    for i in range(0, len(X)-1):
        sum = sum + W[i]*phi(np.abs(e-X[i]), X[len(X)-1])
    return sum


def radial_cost(X):
    cost = 0
    for k in range(0, len(p2.K)):
        cost = cost + (p2.f(p2.K[k])-radial_basis(X, p2.K[k]))**2
    return (p2.b-p2.a)/p2.N*cost


def setup(X):
    M = np.array([[phi(abs(X[i]-X[j]), X[len(X)-1]) for j in range(len(X)-1)] for i in range(len(X)-1)])
    return M


def grad_descent(nodes, L, hyp1, hyp2, iters):
    gradient = grad(cost)
    phi = cost(nodes)
    for k in range(iters):
        g = gradient(nodes)
        #print('count')
        for t in range(iters):
            X = nodes
            x2 = X - 1/L*g
            phi2 = cost(x2)
            if phi2 <= (phi + np.dot(g, x2-X) + L/2*np.linalg.norm(x2-X, 2)):
                nodes = (x2)
                phi = phi2
                L = hyp2*L
                break
            else:
                L = hyp1*L
    return nodes


def grad_descent2(nodes, L, hyp1, hyp2, iters):
    gradient = grad(radial_cost)
    phi = radial_cost(nodes)
    for k in range(iters):
        g = gradient(nodes)
        print('count')
        for t in range(iters):
            X = nodes
            x2 = X - 1/L*g
            phi2 = radial_cost(x2)
            if phi2 <= (phi + np.dot(g, x2-X) + L/2*np.linalg.norm(x2-X, 2)):
                nodes = (x2)
                phi = phi2
                L = hyp2*L
                break
            else:
                L = hyp1*L
    return nodes


nodes = equidist(0, 1, 20)
X = np.append(nodes, 1.)
nodes2 = equidist(0, 1, 1000)

#p2()
#W = weight(X)
#GD = grad_descent2(X, 1., 2., 0.5, 50)
#print(GD)
#print(radial_cost(GD))

print(interpolate_piecewise(0, np.pi/4, 3, 1, 1, G))



#poly, A = interpolate2(g, 0, 1, 1000)
#plt.plot(A, poly)
#plt.show()

#lol = expError2(100, approxInfNorm, 5)
#plt.semilogy(lol, "r")
#plt.show()

#errors, theoretical = error_vs_n(30, F, approxInfNorm)
#errors2, theoretical2 = error_vs_n(40, F, approx2norm)

#G_errors = expError(15, G, approxInfNorm)
#G_errors2 = expError(30, approx2norm)

#plt.semilogy(G_errors, "r")
#plt.semilogy(G_errors2, "b")
#plt.show()

#plt.semilogy(errors, "r")
#plt.semilogy(errors2, "b")
#plt.semilogy(theoretical)
#plt.show()


#y, A = interpolate(C, runge(C), -5, 5, 10000)
#plt.plot(A, y)
#plt.show()

#print(piecewiseLaGrange(0, 1, 10, 0.42, 3, F))
