from __future__ import print_function, division

from sympy.core.add import Add
from sympy.integrals import integrate
from sympy.core.singleton import S
from sympy.core.compatibility import xrange
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.core.symbol import Symbol
from sympy.concrete.summations import Sum


def series_fourier_expansion(expr, x, n, interval_inf, interval_sup):
    """ Returns a truncated Fourier sine and cosine series expansion
    up to n+1 terms (sin(nx)/cos(nx)) over the indicated interval
    """
    # http://mathworld.wolfram.com/FourierSeries.html
    if interval_sup < interval_inf:
        raise ValueError('interval_sup cannot be less than interval_inf')
    L = interval_sup - interval_inf
    l = list()
    integrate_tuple = (x, interval_inf, interval_sup)
    l.append(integrate(expr, integrate_tuple)/L)  # A0/2
    for m in xrange(1, n+1):
        cos_expr = cos(2*m*S.Pi*x/L)
        sin_expr = sin(2*m*S.Pi*x/L)
        An = 2*integrate(expr*cos_expr, integrate_tuple)/L
        Bn = 2*integrate(expr*sin_expr, integrate_tuple)/L
        l.append(An*cos_expr)
        l.append(Bn*sin_expr)
    return Add(*l)


def series_fourier(expr, x, interval_inf, interval_sup):
    """ Returns the unevaluated Fourier sine and cosine series calculated
    over the indicated interval
    """
    # http://mathworld.wolfram.com/FourierSeries.html
    if interval_sup < interval_inf:
        raise ValueError('interval_sup cannot be less than interval_inf')
    n = Symbol('n', integer=True, positive=True)
    L = interval_sup - interval_inf
    integrate_tuple = (x, interval_inf, interval_sup)
    result = integrate(expr, integrate_tuple)/L  # A0/2
    cos_expr = cos(2*n*S.Pi*x/L)
    sin_expr = sin(2*n*S.Pi*x/L)
    An = (2*integrate(expr*cos_expr, integrate_tuple)/L).simplify()
    Bn = (2*integrate(expr*sin_expr, integrate_tuple)/L).simplify()
    if An != S.Zero:
        result += Sum(An*cos_expr, (n, 1, S.Infinity))
    if Bn != S.Zero:
        result += Sum(Bn*sin_expr, (n, 1, S.Infinity))
    return result
