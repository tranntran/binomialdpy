
from scipy.optimize import minimize
from binomialdpy import pvalue

def lower(alpha, sample, n, b, q):
    """
    Calculate lower bound for the one-sided confidence interval for the success probability.

    :param alpha: The confidence level of the returned confidence intervals
    :type alpha: float
    :param sample: Array of binomial samples with Tulap noise.
    :type sample: array
    :param n: Number of trials in Binomial distribution.
    :type n: int
    :param b: Discrete Laplace noise parameter (exp(-&epsilon;)).
    :type b: float
    :param q: Truncated quantile.
    :type q: float
    """
    ans = []
    for s in sample:
        def ci_obj(p):
            tmp = pvalue.right(sample = [s], n = n, p = p, b = b, q = q)
            ans = (tmp[0]-alpha)**2
            return ans
        tmp1 = minimize(ci_obj, 0.5, method='L-BFGS-B', bounds=((0,1),))
        ans.append(tmp1.x[0])
    return ans

def upper(alpha, sample, n, b, q):
    """
    Calculate upper bound for the one-sided confidence interval for the success probability.

    :param alpha: The confidence level of the returned confidence intervals
    :type alpha: float
    :param sample: Array of binomial samples with Tulap noise.
    :type sample: array
    :param n: Number of trials in Binomial distribution.
    :type n: int
    :param b: Discrete Laplace noise parameter (exp(-&epsilon;)).
    :type b: float
    :param q: Truncated quantile.
    :type q: float
    """
    ans = []
    alpha = 1-alpha
    for s in sample:
        def ci_obj(p):
            tmp = pvalue.right(sample = [s], n = n, p = p, b = b, q = q)
            ans = (tmp[0]-alpha)**2
            return ans
        tmp1 = minimize(ci_obj, 0.5, method='L-BFGS-B', bounds=((0,1),))
        ans.append(tmp1.x[0])
    return alpha


def two_side(alpha, sample, n, b, q):
    ans = []
    for s in sample:
        mle = s/n
        mle = max(min([mle, 1]), 0)

        def ci_obj2(p):
            tmp = pvalue.two_side(sample = [s], p = p, n = n, b = b, q = q)
            return tmp[0] - alpha
        
        if mle > 0:
            tmp = minimize(ci_obj2, mle/2, method='L-BFGS-B', bounds=((0, mle),))
            L = tmp.x
        else:
            L = 0
        
        if mle < 1:
            tmp = minimize(ci_obj2, (1-mle)/2, method='L-BFGS-B', bounds=((mle, 1),))
            U = tmp.x
        else:
            U = 1
        ans.append((L[0], U[0]))
    return ans      