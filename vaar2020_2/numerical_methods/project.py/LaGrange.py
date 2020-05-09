# LaGrange interpolation
# inputs: n nodes: x_i and n values: y_j, interval bounds: a,b
import math
import numpy as np
from autograd import grad
from numpy import linalg as LA
import matplotlib.pyplot as plt

def LaGrange(X, Y):
    L = np.poly1d([0])
    for k in range(0, len(X)):
        p = np.poly1d([1])
        for i in range(0, len(X)):
            if not i == k:
                p = np.polymul(p, np.poly1d([1 / (X[k] - X[i]), -X[i] / (X[k] - X[i])]))
        L = np.polyadd(L, np.polymul(Y[k], p))
    return(L)


#L = LaGrange([1, 2, 3], [1, 4, 9])
#print(L)
#print(L(5))


# A is the x values for which we want to plot.
def interpolate(X, Y, start, end, iters):
    L = LaGrange(X, Y)
    A = equidist(start, end, iters)
    vals = []
    i = 0
    while i < iters:
        vals.append((L(A[i])))
        i += 1
    return vals


def LaGrange2(X, Y, x):
    n = len(X)
    L = []
    terms = []
    for k in range(0, n):
        factors = []
        for i in range(0, n):
            if (X[i] != X[k]):
                factors.append((x-X[i])/(X[k]-X[i]))
        product = math.prod(factors)
        L.append(product)
        terms.append(L[k]*Y[k])

    p = math.fsum(terms)
    return p


def piecewiseLaGrange(a, b, K, x, n, f):
    i = a
    X = []
    p = np.inf
    while i < b:
        X.append([i, i+(b-a)/K])
        i += (b-a)/K
    #print(X)
    for y in X:
        #print(y)
        if x < y[1]:
            p = LaGrange2(equidist(y[0], y[1], n), func_values(f, equidist(y[0], y[1], n)), x)
            #print(p)
            break
    return p


# A is the x values for which we want to plot.
def interpolate3(start, end, iters, K, n, f):
    A = equidist(start, end, iters)
    vals = []
    i = 0
    while i < iters:
        vals.append(piecewiseLaGrange(start, end, K, A[i], n, f))
        i += 1
    return vals

# a,b interval bounds, n number of nodes
def equidist(a, b, n):
    X = []
    for k in range(0,n):
        X.append(a+k*((b-a)/n))
    return X


def chebyshev(a, b, n):
    X = []
    for k in range(1, n+1):
        X.append(1/2*(a+b)+1/2*(b-a)*math.cos((2*k-1)/(2*n)*math.pi))
    return X


def runge(X):
    Y = []
    for x in X:
        Y.append( 1/(x*x + 1) )
    return Y


# A is the x values for which we want to plot.
def interpolate2(X, Y, start, end, iters):
    A = equidist(start, end, iters)
    vals = []
    i = 0
    while i < iters:
        vals.append(LaGrange2(X, Y, A[i]))
        i += 1
    return vals


def approxInfNorm(fx, p_n):
    max = np.linalg.norm(abs(fx - p_n), np.inf)
    return max


def approx2norm(fx, p_n):
    norm = np.linalg.norm(abs(fx-p_n), 2)
    return norm


def func_values(f, X):
    fx = []
    for x in X:
        fx.append(f(x))
    fx = np.asarray(fx)
    return fx


def F(x):
    return math.cos(2 * math.pi * x)


def G(x):
    return math.exp(3*x)*math.sin(2*x)


def error_vs_n(N, norm):
    i = 1
    list = []
    theoretic = []
    while i < N:
        X = equidist(0, 1, 100 * i)
        fx = func_values(F, X)
        interpol_vals = func_values(F, chebyshev(0, 1, i))
        p_n = interpolate2(chebyshev(0, 1, i), interpol_vals, 0, 1, 100 * i)
        list.append( norm( fx, p_n ) )
        theoretic.append( (2*math.pi)**(i+1)/(math.factorial(i+1)) )
        i += 1
    return list, theoretic


def expError(N, norm):
    i = 1
    list = []
    while i < N:
        X = equidist(0, math.pi/4, 100*i)
        fx = func_values(G, X)
        interpol_vals = func_values(G, chebyshev(0, math.pi/4, i))
        p_n = interpolate2(chebyshev(0,1, i), interpol_vals, 0, 1, 100*i)
        list.append(norm(fx, p_n))
        i += 1
    return list


def expError2(K, norm, n, bla):
    i = 1
    list = []
    X = equidist(0, math.pi / 4, 100*n)
    while i < K:
        fx = func_values(G, X)
        p_n = interpolate3(0, math.pi/4, 100*n, i, n, G)
        list.append(norm(fx, p_n))
        i += 1
    return list


#lol = expError2(1000, approxInfNorm, 3)
#plt.semilogy(lol, "r")
#plt.show()

#errors, theoretical = error_vs_n(40, approxInfNorm)
#errors2, theoretical2 = error_vs_n(40, approx2norm)

G_errors = expError(40, approxInfNorm)
G_errors2 = expError(40, approx2norm)

plt.semilogy(G_errors, "r")
plt.semilogy(G_errors2, "b")
plt.show()

#plt.semilogy(errors, "r")
#plt.semilogy(errors2, "b")
#plt.semilogy(theoretical)
#plt.show()

C = chebyshev(-5,5,10)

#y, A = interpolate(C, runge(C), -5, 5, 10000)
#plt.plot(A, y)
#plt.show()

#print(piecewiseLaGrange(0, 1, 10, 0.42, 3, F))



def costfunction(f, a, b, X):
    N = 1000
    CX = 0
    A = equidist(a, b, N)
    fmerket = [val for (key, val) in f]
    for x in A:
        p_n_x = LaGrange2(X, func_values(F, X), x)
        CX = CX + ( f[x]-p_n_x )**2
    CX = CX*((b-a)/N)
    return CX


A = equidist(0, 1, 1000)
def interpolcreate(X):
    p_n = []
    f = []
    for x in A:
        f.append(F(x))
        p_n.append(LaGrange2(X, func_values(F, X), x))

    return p_n, f


X = chebyshev(0, 1, 7)
a = 0
b = 1
n = 7
p_n, f = interpolcreate(chebyshev(0, 1, n))

p_n = np.array(p_n)
#f = np.array(f)


from functools import partial #Denne gjør mye morsomt

#Det første du må gjøre er å skrive om costfunction til costfunction(f, a, b, X)

#Antatt at man har verdier for f, start a og slutt b
f, a, b = f, 0, 1 #Bytt gjerne disse til de riktige verdiene

differentiableCost = partial(costfunction, f, a, b)
costDerived = grad(differentiableCost)

print(costDerived(X)) #Prøver å printe gradienten med en X-verdi



# The cost function takes as input X - the interpolation nodes,
# returning C(X) -- the cost of interpolating with those nodes.
def cost(X):
    CX = 0
    i = 0
    while i < 1000:
        CX = CX + ((b-a)/1000)*((f[i]-p_n[i])**2)
        i = i + 1
    return CX


def terms():
    terms = []
    for i in range(0, 1000):
        terms.append(((f[i]-p_n[i])**2))
    return terms


terms = terms()


def LaGrange3(X, Y, x):
    n = len(X)
    L = []
    terms = []
    for k in range(0, n):
        factors = []
        for i in range(0, n):
            if (X[i] != X[k]):
                factors.append((x-X[i])/(X[k]-X[i]))
        product = math.prod(factors)
        L.append(product)
        terms.append(L[k]*Y[k])

    p = math.fsum(terms)
    return p


def cost3(X, f):
    n = len(X)
    L = []
    terms = []
    for k in range(0, n):
        factors = []
        for i in range(0, n):
            if (X[i] != X[k]):
                factors.append((x - X[i]) / (X[k] - X[i]))
        product = math.prod(factors)
        L.append(product)
        terms.append(L[k] * f(X[k]))
    p = math.fsum(terms)


def cost2(X):
    return ((b-a)/1000)*np.sum(terms)


print(grad(cost)(0.34))

dc = grad(costfunction)
DC = grad(cost)
DC2 = grad(cost2)
#print(DC)
#print(DC2)
#print(dc)


cost = cost(chebyshev(0, 1, n))
#print(cost)

#print(chebyshev(0, 1, 7))

### Max sin kode:
def langrange_basis_product(values):
    return np.product(values)


def lagrange_basis(x, x_list, j, k):
    lagrange_basis_summands = []
    for m in range(k):
        if m != j:
            lagrange_basis_summands.append((x - x_list[m]) / (x_list[j] - x_list[m]))
    return langrange_basis_product(lagrange_basis_summands)


def lagrange_sum_function(x_val, y_val):
    k = len(x_val)
    return lambda x: sum(y_val[j]*lagrange_basis(x, x_val, j, k) for j in range(k))


def lagrange_interpolated_polynomial(nodes):
    x_values, y_values = [x for (x,_) in nodes], [y for (_,y) in nodes]
    polynomial = lagrange_sum_function(x_values, y_values)
    return polynomial


def lagrange_interpolation_with_eval(nodes, point):
    """
    -- Parameters --
    them_nodes: Array with nodes
    point:
    -- Returns --
    The polynomial evaluated in the wanted point
    """
    P = lagrange_interpolated_polynomial(nodes)
    return P(point)


def lagrange_interpolation_degree(nodes):
    return len(nodes) - 1

nodes1 = func_values(F, equidist(0,1,5))
nodes2 = func_values(F, equidist(0,1, 1000))

#print(lagrange_interpolation_with_eval(nodes, 0.445))


def p():
    """
    f er funksjon
    a er start
    b er slutt
    N er antall kjente noder
    known er de kjente nodene med elementer på formen (x,f(x))
    """
    p.f, p.a, p.b, p.N, p.k = F, 0., 1., 1000, nodes2


def cost6(X):
        q = partial(lagrange_interpolation_with_eval,[(x, p.f(x)) for x,_ in X])
        return (p.b-p.a)/p.N*np.sum([(p.f(k) - q(k))**2 for k,fk in p.k])

#p()
#cost6(nodes1)
#print(grad(cost6)(nodes1))

from autograd import jacobian


def polymul(P,Q):
    produkt=np.zeros(len(Q)+len(P))
    for p in range(len(P)):
        for q in range(len(Q)):
            produkt[p+q]=(produkt[p+q])+((P[p])*(Q[q]))
    return produkt

def K(x):
    return x**2

def LaGrange3(X): #Dette er en funksjon som lager et Lagrange polynom med noder
    n = len(X)
    L = np.poly1d([0])
    for i in range(n):
        pn = np.poly1d(X[i][1])
        for j in range(n):
            if (X[i][0] != X[j][0]):
                print((X[i][0])-X[j][0])
                factor = X[i][0] - X[j][0]
                pn = polymul(pn, [1, -X[j][0]] / factor)
        L = np.polyadd(L, pn)
    return L

#print( Lagrange3([[1,1], [2,4], [3,9]]))
def LaGrange4(X,x):
    n = len(X)
    L = []
    terms = []
    np.array(L)
    np.array(terms)
    for k in range(n):
        factors = [(x - X[i][0]) / (X[k][0] - X[i][0]) for i in range(n) if (X[i][0] != X[k][0])]
        L.append(np.product(factors))
        terms.append(L[k] * X[k][1])
    return lambda x: sum(terms)


def eval(X,x):
    return LaGrange4(X, x)(x)

nodes = func_values2(F, equidist(0,1,5))
nodes2 = func_values2(F, equidist(0,1, 1000))


def p():
    """
    f er funksjon
    a er start
    b er slutt
    N er antall kjente noder
    known er de kjente nodene med elementer på formen (x,f(x))
    """
    p.f, p.a, p.b, p.N, p.k = F, 0., 1., 1000, nodes2


def cost(X):
        q = partial(eval,[[x, p.f(x)] for x,_ in X])
        return (p.b-p.a)/p.N*np.sum([(p.f(k) - q(k))**2 for k,fk in p.k])

p()
print(eval(nodes, 0.45))
print(cost(nodes))


gradient = grad(cost)
print((gradient(nodes)))


[array([0.00389171, 0.32522667, 0.21627479, 0.78839504, 0.67916447]),
 array([-0.26904088, -0.37753357, -0.02683152,  0.72720619,  1.04672845]),
 array([-0.26904088, -0.37753357, -0.02683152,  0.72720619,  1.04672845]),
 array([-2.60340254e-01, -3.71679650e-01, -4.26691803e-04,  1.14512890e+00, 9.46502481e-01]),
 array([-0.16467191, -0.30257144,  0.21307774,  1.10941211,  0.84994083]),
 array([-0.23099247, -0.35439457, -0.07313052,  1.11982104,  0.9538059 ]),
 array([-1.37958038e-01, -2.92910494e-01,  3.08444837e-04,  1.05925841e+00, 9.17131065e-01]),
 array([-0.17489039, -0.31992967, -0.04498068,  1.08152928,  0.98461394]),
 array([-0.07388777, -0.25206966,  0.10472432,  1.04612501,  0.92388266])]
[1.4651723699319987, 0.5002873093637814, 0.5002873093637814, 0.16321002324461625, 0.5967256796059334, 0.3619461743894756, 0.017396852564960808, 0.10909198569873729, 0.3077712338246194]



[array([0.00389171, 0.32522667, 0.21627479, 0.78839504, 0.67916447]),
 array([-0.26904088, -0.37753357, -0.02683152,  0.72720619,  1.04672845]),
 array([-0.26904088, -0.37753357, -0.02683152,  0.72720619,  1.04672845]),
 array([-2.60340254e-01, -3.71679650e-01, -4.26691803e-04,  1.14512890e+00,
        9.46502481e-01]),
 array([-0.16467191, -0.30257144,  0.21307774,  1.10941211,  0.84994083]),
 array([-0.23099247, -0.35439457, -0.07313052,  1.11982104,  0.9538059 ]),
 array([-1.37958038e-01, -2.92910494e-01,  3.08444837e-04,  1.05925841e+00,
        9.17131065e-01]),
 array([-0.17489039, -0.31992967, -0.04498068,  1.08152928,  0.98461394]),
 array([-0.07388777, -0.25206966,  0.10472432,  1.04612501,  0.92388266]),
 array([-0.18467574, -0.3279511 , -0.07874089,  1.0668791 ,  0.97980548]),
 array([-0.04064007, -0.2329607 ,  0.12155321,  1.00378094,  0.88190297]),
 array([-0.156041  , -0.30972786, -0.06503078,  1.03338307,  0.95822299]),
 array([-0.08345597, -0.26360994,  0.03325508,  1.00581304,  0.92022374]),
 array([-0.16524231, -0.31776134, -0.07759985,  1.04033423,  0.9814398 ]),
 array([-0.05087979, -0.24423523,  0.07326021,  0.98969486,  0.91363812]),
 array([-0.16273629, -0.3170732 , -0.08457531,  1.02860684,  0.98026007]),
 array([-0.04888655, -0.24473159,  0.06163493,  0.97574855,  0.91361355]),
 array([-0.15877637, -0.31547014, -0.08732544,  1.02018857,  0.98178013]),
 array([-0.04842975, -0.24600328,  0.05158817,  0.96818711,  0.91942212])]
[1.4651723699319987, 0.5002873093637814, 0.5002873093637814, 0.16321002324461625, 0.5967256796059334,
 0.3619461743894756, 0.017396852564960808, 0.10909198569873729, 0.3077712338246194, 0.21648716861005585,
 0.5044308904380891, 0.06687305390500166, 0.13398742397721294, 0.13723355821776678, 0.30913769139538283,
 0.1374491864070939, 0.2928991597483549, 0.12930331723911231, 0.2682393850344777]

sup1node = expError2(100, approxInfNorm, 1, equidistant)
sup2node = expError2(100, approxInfNorm, 2, equidistant)
sup3node = expError2(100, approxInfNorm, 3, equidistant)
sup4node = expError2(100, approxInfNorm, 4, equidistant)
sup5node = expError2(100, approxInfNorm, 5, equidistant)
sup6node = expError2(100, approxInfNorm, 6, equidistant)
sup7node = expError2(100, approxInfNorm, 7, equidistant)
sup8node = expError2(100, approxInfNorm, 8, equidistant)
sup9node = expError2(100, approxInfNorm, 9, equidistant)
sup10node = expError2(100, approxInfNorm, 10, equidistant)
plt.semilogy(sup1node, label = 'sup norm 1 node')
plt.semilogy(sup2ode, label = 'sup norm 1 node')
plt.semilogy(sup3ode, label = 'sup norm 1 node')
plt.semilogy(sup4ode, label = 'sup norm 1 node')
plt.semilogy(sup5ode, label = 'sup norm 1 node')
plt.semilogy(sup6ode, label = 'sup norm 1 node')
plt.semilogy(sup7ode, label = 'sup norm 1 node')
plt.semilogy(sup8ode, label = 'sup norm 1 node')
plt.semilogy(sup9ode, label = 'sup norm 1 node')
plt.semilogy(sup10nde, label = 'sup norm 1 node')
plt.semilogy(sup1node, label = 'sup norm 1 node')
plt.ylabel('error equidistant nodes')
plt.xlabel('number of subintervals')
plt.legend()
plt.show()