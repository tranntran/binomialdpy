from scipy.stats import binom
import numpy as np

def right(sample, n, p, b, q):
    """
    Calculate UMP right-sided p-values for binomial data under (&epsilon;, &delta;)-DP.

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
    Calculate UMP right-sided p-values for binomial data under (&epsilon;, &delta;)-DP.

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