from scipy.stats import binom
import numpy as np
from . import tulap

def right(sample, n, p, b, q):
    """
    Calculate UMP right-sided p-values for binomial data.

    :param sample: Array of binomial samples with Tulap noise.
    :type sample: array
    :param n: Number of trials in Binomial distribution.
    :type n: int
    :param p: Success probability for each trial.
    :type p: float
    :param b: Discrete Laplace noise parameter (exp(-&epsilon;)).
    :type b: float
    :param q: Truncated quantile.
    :type q: float
    """

    reps = len(sample)
    pval = [0 for _ in range(reps)]
    values = [i for i in range(n+1)]

    B = binom.pmf(values, n = n, p = p)
    
    for r in range(reps):
        F = [tulap.cdf(t = x-sample[r], m = 0, b = b, q = q) for x in values]
        pval[r] = np.dot(F, B)

    return pval


def left(sample, n, p, b, q):
    """
    Calculate UMP right-sided p-values for binomial data.

    :param sample: Array of binomial samples with Tulap noise.
    :type sample: array
    :param n: Number of trials in Binomial distribution.
    :type n: int
    :param p: Success probability for each trial.
    :type p: float
    :param b: Discrete Laplace noise parameter (exp(-&epsilon;)).
    :type b: float
    :param q: Truncated quantile.
    :type q: float
    """

    reps = len(sample)
    pval = [0 for _ in range(reps)]
    values = [i for i in range(n+1)]

    B = binom.pmf(values, n = n, p = p)
    
    for r in range(reps):
        F = [1 - tulap.cdf(t = x-sample[r], m = 0, b = b, q = q) for x in values]
        pval[r] = np.dot(F, B)

    return pval


def two_side(sample, n, p, b, q):
    """
    Calculating asymptotically unbiased DP two-sided p-value for binomial data.

    :param sample: Array of binomial samples with Tulap noise.
    :type sample: array
    :param n: Number of trials in Binomial distribution.
    :type n: int
    :param p: Success probability for each trial.
    :type p: float
    :param b: Discrete Laplace noise parameter (exp(-&epsilon;)).
    :type b: float
    :param q: Truncated quantile.
    :type q: float
    """
    reps = len(sample)
    T = [abs(x-n*p) for x in sample]
    tmp1 = np.asarray(right(sample = [x+n*p for x in T], n = n, p = p, b = b, q = q))
    tmp2 = np.subtract([1]*reps, right(sample = [n*p-x for x in T], n = n, p = p, b = b, q = q))
    ans = tmp1 + tmp2
    return list(ans)