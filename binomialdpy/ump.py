import math
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
from tulap import cdf

def left(p, n, alpha, epsilon, delta):
    """
    Calculating simple and left-sided DP-UMP tests under (&epsilon;, &delta;)-DP.

    :param p: The success probability for each trial.
    :type sample: float
    :param n: The number of trials in Binomial distribution.
    :type n: int
    :param alpha: The alpha level of the tests.
    :type alpha: float
    :param epsilon: Privacy parameter in (&epsilon;, &delta;)-DP.
    :type epsilon: float
    :param delta: Privacy parameter in (&epsilon;, &delta;)-DP.
    :type delta: float
    """
    b = math.exp(-epsilon)
    q = 2*delta*b/(1-b+2*delta*b)
    values = [i for i in range(n+1)]

    B = binom.pmf(values, n = n, p = p)

    def obj(s):
        phi = [cdf(t = x-s, m = 0, b = b, q = q) for x in values]
        return np.dot(B, phi) - alpha
    
    lower = -1
    while obj(lower) < 0:
        lower *= 2
    upper = 1
    while obj(upper) > 0:
        upper *= 2
    
    root = brentq(obj, lower, upper)
    phi = [cdf(t = x-root, m = 0, b = b, q = q) for x in values]
    return phi


def right(p, n, alpha, epsilon, delta):
    """
    Calculating simple and right-sided DP-UMP tests under (&epsilon;, &delta;)-DP.

    :param p: The success probability for each trial.
    :type sample: float
    :param n: The number of trials in Binomial distribution.
    :type n: int
    :param alpha: The alpha level of the tests.
    :type alpha: float
    :param epsilon: Privacy parameter in (&epsilon;, &delta;)-DP.
    :type epsilon: float
    :param delta: Privacy parameter in (&epsilon;, &delta;)-DP.
    :type delta: float
    """
    b = math.exp(-epsilon)
    q = 2*delta*b/(1-b+2*delta*b)
    values = [i for i in range(n+1)]

    B = binom.pmf(values, n = n, p = p)

    def obj(s):
        phi = [cdf(t = x-s, m = 0, b = b, q = q) for x in values]
        return np.dot(B, phi) - alpha
    
    lower = -1
    while obj(lower) < 0:
        lower *= 2
    upper = 1
    while obj(upper) > 0:
        upper *= 2
    
    root = brentq(obj, lower, upper)
    phi = [1-cdf(t = x-root, m = 0, b = b, q = q) for x in values]
    return phi