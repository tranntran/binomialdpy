#' Calculating Unbiased Two-Sided DP-UMP Tests
#' @aliases twoside
#' @aliases umpu
#'
#' @param p The success probability for each trial (parameter \eqn{\p}
#'   in Binomial(n, \eqn{\p}))
#' @param n The number of trials in Binomial distribution (parameter n in
#'   Binomial(n, \eqn{\p}))
#' @param alpha Level of the tests
#' @param epsilon Parameter \eqn{\epsilon} in \eqn{(\epsilon, \delta)}-DP
#' @param delta Parameter \eqn{\delta} in \eqn{(\epsilon, \delta)}-DP
#'
#' @return A vector of unbiased two-sided DP-UMP tests
#' @export
#' @references Awan, Jordan Alexander, and Aleksandra Slavkovic. 2020.
#'   "Differentially Private Inference for Binomial Data". Journal of Privacy
#'   and Confidentiality 10 (1). \url{https://doi.org/10.29012/jpc.725}.
#' @seealso Calculating simple and one-sided DP-UMP tests (\code{\link{umpLeft}}
#'   or \code{\link{umpRight}}) and asymptotically unbiased two-sided DP-UMP
#'   tests (\code{\link{umpuApprox}})
#' @examples
#' #Comparing unbiased DP-UMP tests obtained by umpuApprox and UMPU
#' asymp <- umpuApprox(p = 0.4, n = 10, alpha = 0.05, epsilon = 1, delta = 0.01)
#' twoside <- UMPU(p = 0.4, n = 10, alpha = 0.05, epsilon = 1, delta = 0.01)
#'
#' #Plot the probability of rejecting the null hypothesis based on x
#' plot(asymp, type = "l", lwd = 1.5, col = "red", xlab = "x",
#'   ylab = "Phi (Probability of rejecting the null hypothesis)",
#'   main = "Unbiased DP-UMP Tests")
#' lines(twoside, type = "l", lwd = 1.5, lty = 2, col = "blue")
#' legend("topleft", legend=c("Asymptotically UMPU", "UMPU"),
#'   col=c("red", "blue"), lty=1:2, lwd = 1.5)
#'
import math
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq

def umpu(p, n, alpha, epsilon, delta):
    """
    Calculating unbiased two-sided DP-UMP tests under (&epsilon;, &delta;)-DP.

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
    BX = np.multiply(B, [x - n*p for x in values])
    s = 0

    def obj(k):
        greater_k = [x >= k for x in values]

        def mini_obj(s):
            F1 = [tulap.cdf(t = x-k-s, m = 0, b = b, q = q) for x in values]
            F2 = [tulap.cdf(t = k-x-s, m = 0, b = b, q = q) for x in values]
            phi = np.add([x if y == True else 0 for x, y in zip(F1, greater_k)], \
                [x if y == False else 0 for x, y in zip(F2, greater_k)])
            ans = np.dot(B, phi) - alpha
            return ans
        
        lower = -1
        while mini_obj(lower) < 0:
            lower *= 2
        upper = 1
        while mini_obj(upper) > 0:
            upper *= 2
        
        mini_root = brentq(mini_obj, lower, upper)
        F1 = [tulap.cdf(t = x-k-mini_root, m = 0, b = b, q = q) for x in values]
        F2 = [tulap.cdf(t = k-x-mini_root, m = 0, b = b, q = q) for x in values]
        phi = np.add([x if y == True else 0 for x, y in zip(F1, greater_k)], \
            [x if y == False else 0 for x, y in zip(F2, greater_k)])

        return np.dot(BX, phi)
    
    lower = -n
    while obj(lower) < 0:
        lower *= 2
    upper = 2*n
    while obj(upper) > 0:
        upper *= 2
    
    k = brentq(obj, lower, upper)
    greater_k = [x >= k for x in values]

    def mini_obj(s):
        F1 = [tulap.cdf(t = x-k-s, m = 0, b = b, q = q) for x in values]
        F2 = [tulap.cdf(t = k-x-s, m = 0, b = b, q = q) for x in values]
        phi = np.add([x if y == True else 0 for x, y in zip(F1, greater_k)], \
            [x if y == False else 0 for x, y in zip(F2, greater_k)])
        ans = np.dot(B, phi) - alpha
        return ans
    
    lower = -1
    while mini_obj(lower) < 0:
        lower *= 2
    upper = 1
    while mini_obj(upper) > 0:
        upper *= 2
    s = brentq(mini_obj, lower, upper)
    F1 = [tulap.cdf(t = x-k-s, m = 0, b = b, q = q) for x in values]
    F2 = [tulap.cdf(t = k-x-s, m = 0, b = b, q = q) for x in values]
    phi = np.add([x if y == True else 0 for x, y in zip(F1, greater_k)], \
        [x if y == False else 0 for x, y in zip(F2, greater_k)])
    return phi