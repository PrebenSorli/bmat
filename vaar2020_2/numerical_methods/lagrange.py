import math
from itertools import product
import autograd.numpy as np
from autograd import grad
from numpy import linalg as LA
import matplotlib.pyplot as plt
from functools import partial


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


def piecewiseLaGrange(a, b, K, x, n, f):
    i = a
    X = []
    while i < b:
        X.append([i, i + (b - a) / K])
        i += (b - a) / K
    for y in X:
        if x <= y[1]:
            return eval(equidist(y[0], y[1], n), f, x)


# A is the x values for which we want to plot.
def interpolate_piecewise(start, end, iters, K, n, f):
    A = equidist(start, end, iters)
    #print(A)
    return [piecewiseLaGrange(start, end, K, A[i], n, f) for i in range(iters)], A


# A is the x values for which we want to plot.
def interpolate2(X, f, start, end, iters):
    A = equidist(start, end, iters)
    return [eval(X, f, A[i]) for i in range(iters)], A


# a,b interval bounds, n number of nodes
def equidist(a, b, n):
    return np.asarray(np.linspace(a, b, n))


def chebyshev(a, b, n):
    return np.asarray([1 / 2 * (a + b) + 1 / 2 * (b - a) * math.cos((2 * k - 1) / (2 * n) * math.pi) for k in range(1, n+1)])


def runge(x):
    return 1/(x**2+1)


def rad_function(x):
    return 3 / 4 *(np.exp((-(9*x - 2)**2) / 4) + (np.exp((-(9*x + 1)**2) / 49))) + 1 / 2 * (np.exp((-(9*x - 7)**2) / 4)) - 1/10 * (np.exp(-(9*x - 4)**2))


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

#print(piecewiseLaGrange(0, np.pi/4, 3, 0.39269908, 1, G))
#p, A = (interpolate2(equidist(0, np.pi/4, 10), G, 0,np.pi/4, 1000))
#p2, A2 = interpolate_piecewise(0,np.pi/4, 10, 1000, 1, G)
#plt.plot(p, A)
#plt.plot(p2, A2)
#plt.show()
#print(interpolate2(equidist(0,np.pi/4, 2), G, 0, np.pi/4, 2))


def error_vs_n(N, f, norm, nodetype):
    M = []
    theoretic = []
    for i in range(1, N):
        X = equidist(0, 1, 100 * i)
        fx = func_values(f, X)
        p_n,_ = interpolate2(nodetype(0, 1, i), f, 0, 1, 100 * i)
        M.append(norm(fx, p_n))
        theoretic.append((2 * np.pi)**(i + 1) / (math.factorial(i + 1)))
    return M, theoretic


def expError(N, f, norm, nodetype):
    M = []
    for i in range(1, N):
        X = equidist(0, np.pi/4, 100 * i)
        fx = func_values(f, X)
        p_n,_ = interpolate2(nodetype(0, np.pi/4, i), f, 0, np.pi/4, 100 * i)
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

print(np.finfo(float).eps)
errors = expError(50, G, approxInfNorm, chebyshev)
errors2 = expError(50, G, approx2norm, chebyshev)

plt.semilogy(errors, "r", label = 'sup error Chebyshev nodes')
plt.semilogy(errors2, "b", label = 'square error Chebyshev nodes')
#plt.semilogy(theoretical, label = 'theoretical error bound')
plt.ylabel("error")
plt.xlabel('number of nodes')
plt.legend()
plt.show()



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
    for k in p.K:
        cost = cost + (p.f(k)-eval(X, p.f, k))**2
    return (p.b-p.a)/p.N*cost


def p2(nodes):
    """
      f er funksjon
      a er start
      b er slutt
      N er antall kjente noder
      K er de kjente nodene med elementer på formen (x)
      """
    p2.f, p2.a, p2.b, p2.N, p2.K, p2.nodes = runge, -1., 1., 1000, nodes2, nodes


def phi(r, epsilon):
    return np.exp(-(epsilon*r)**2)


def weight(X):
    #Y = np.array(X[:-1])
    M = np.array([[phi(abs(X[i] - X[j]), X[len(X) - 1]) for j in range(len(X) - 1)] for i in range(len(X) - 1)])
    #print(M)
    #print(p2.nodes)
    W = np.linalg.solve(M, func_values(p2.f, p2.nodes))
    return W


def radial_basis(X, e):
    sum = 0
    W = weight(X)
    for i in range(0, len(X)-1):
        sum = sum + W[i]*phi(np.abs(e-X[i]), X[len(X)-1])
    return sum


def interpolate3(X, start, end, iters):
    #X = np.append(X, np.inf)
    A = equidist(start, end, iters)
    return [radial_basis(X, A[i]) for i in range(iters)], A


def radial_cost(X):
    cost = 0
    for k in p2.K:
        cost = cost + (p2.f(k)-radial_basis(X, k))**2
    return (p2.b-p2.a)/p2.N*cost


def grad_descent(nodes, L, hyp1, hyp2, iters):
    gradient = grad(cost)
    phi = cost(nodes)
    for k in range(iters):
        g = gradient(nodes)
        #print('count')
        for t in range(iters*50):
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
    C = []
    for k in range(iters):
        g = gradient(nodes)
        C.append(radial_cost(nodes))
        for t in range(20):
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
    return nodes, C

'''
nodes = equidist(-1, 1, 3)
X = np.append(nodes, 3.)
nodes2 = equidist(-1, 1, 1000)

nodes3 = equidist(-1,1,1)
'''
def grad2_error_vs_n(N, f, norm):
    M = []
    for i in range(2, N):
        p2(equidist(-1,1,i))
        #print(p2.nodes)
        X = np.append(p2.nodes, 3.)
        X2 = equidist(-1, 1, 1000)
        fx = func_values(f, X2)
        g, Co = grad_descent2(X, 100, 2., 0.9, 40)
        p_n,_ = interpolate3(g, -1, 1, 1000)
        M.append(norm(fx, p_n))
    return M


def rbf_error_vs_n(N, f, norm, nodetype):
    M = []
    for i in range(2, N):
        p2(nodetype(-1,1,i))
        X2 = equidist(-1,1,1000)
        X = np.append(p2.nodes, 3.)
        fx = func_values(f,X2)
        p_n,_ = interpolate3(X, -1,1,1000)
        M.append(norm(fx, p_n))
    return M
'''
errorEqui = rbf_error_vs_n(4, runge, approx2norm, equidist)
errorCheby = rbf_error_vs_n(4, runge, approx2norm, chebyshev)
errorGrad2 = grad2_error_vs_n(4, runge, approx2norm)

#g, errorGrad2 = grad_descent2(X, 100, 2., 0.9, 6)

plt.semilogy(errorEqui, label = 'Equidistant nodes, runge function')
plt.semilogy(errorGrad2, label = 'Gradient descent nodes, runge function')
plt.semilogy(errorCheby, label = 'Chebyshev nodes, runge function')
plt.xlabel("number of nodes")
plt.ylabel("square error")
plt.legend()
plt.show()
'''
'''
print(equidist(0, 1, 10))
print(np.linspace(0,1, 10))
rungeplotEq,_ = interpolate2(equidist(-5, 5, 10), runge, -5., 5., 1000)
rungeplotChe,A = interpolate2(chebyshev(-5, 5, 10), runge, -5, 5, 1000)
plt.plot(A, rungeplotEq, label = 'Equidistant nodes')
plt.plot(A, rungeplotChe, label = 'Chebyshev nodes')
plt.legend()
plt.show()
'''
