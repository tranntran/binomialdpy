import numpy as np
from scipy.stats import norm
import math

def random(n, m = 0, b = 0, q = 0):
  """
  Generate random sample from the Tulap distribution.

  :param n: Number of observations.
  :type n: int
  :param m: Array of medians.
  :type m: array
  :param b: Array of Discrete Laplace noise parameters (exp(-&epsilon;)).
  :type b: array
  :param q: Array of truncated quantiles.
  :type q: array
  """
  if (q >= 0): 
    alpha = .95
    lcut = q/2
    rcut = q/2

    def approx_trials(n, prob = 1, alpha = 0):
        a = prob**2
        b = -((2 * n * prob) + ((norm.ppf(alpha)**2) * prob * (1 - prob)))
        c = n^2
        return round(-b + math.sqrt(b**2 - (4 * a * c))/ (2 * a), 0)

    # Calculate actual amount needed
    q = lcut + rcut
    n2 = int(approx_trials(n, prob=(1 - q), alpha=alpha))

    # Sample from the original Tulambda distribution
    geos1 = np.random.geometric(1-b, n2)
    geos2 = np.random.geometric(1-b, n2)
    unifs = np.random.uniform(-0.5, 0.5, n2)
    samples = m + geos1 - geos2 + unifs

    # Cut the tails based on the untampered CDF (ie no cuts)
    probs = [cdf(x, m = m, b = b) for x in samples]
    is_mid = [(lcut <= x) & (x <= (1 - rcut)) for x in probs]

    # Abuse the NA property of R wrt arithmetics
    mids = samples[is_mid]
    while len(mids) < n:
      diff = n - len
      mids = mids.append(random(diff, m=m, b=b,q=q))
    return mids[0:n]

  geos1 = np.random.geometric(1-b, n)
  geos2 = np.random.geometric(1-b, n)
  unifs = np.random.uniform(-0.5, 0.5, n)
  samples = m + geos1 - geos2 + unifs
  return(samples)


def cdf(t, m = 0, b = 0, q = 0):
  """
  Give the cummulative distribution function (CDF) from the Tulap distribution.

  :param t: Array of quantiles.
  :type t: array
  :param m: Array of medians.
  :type m: array
  :param b: Array of Discrete Laplace noise parameters (exp(-&epsilon;)).
  :type b: array
  :param q: Array of truncated quantiles.
  :type q: array
  """

  lcut = q/2
  rcut = q/2
  # Normalize
  t = t - m

  # Split the positive and negsative t calculations, and factor out stuff
  r = round(t)
  try:
    g = -math.log(b)
  except ValueError:
    g = -float("inf")
  try:
    l = math.log(1 + b)
  except ValueError:
    l = float("inf")
  k = 1 - b
  negs = math.exp((r * g) - l + math.log(b + ((t - r + (1/2)) * k)))
  poss = 1 - math.exp((r * (-g)) - l + math.log(b + ((r - t + (1/2)) * k)))

  # Check for infinities
  if math.isinf(negs): negs = 0
  if math.isinf(poss): poss = 0

  # Truncate wrt the indicator on t's positivity
  is_leq0 = 1 if t <= 0 else 0
  trunc = (is_leq0 * negs) + ((1 - is_leq0) * poss)

  # Handle the cut adjustment and scaling
  q = lcut + rcut
  is_mid = 1 if ((lcut <= trunc) & (trunc <= (1 - rcut))) else 0
  is_rhs = 1 if (1 - rcut) < trunc else 0
  return (((trunc - lcut) / (1 - q)) * is_mid + is_rhs)
