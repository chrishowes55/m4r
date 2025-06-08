from scipy import stats, linalg
import numpy as np
import pandas as pd
import os
from scipy.special import logsumexp, gammaln, logit, softmax
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns   
from scipy.stats import invweibull
from scipy.integrate import quad

def generate_next_timepoint(m, k, w, mu, gamma, nu, zeta, S, dt, rng=None):
    """
    Simulates the transitions between the homozygous demtheylated, heterozygous
    and homozygous methylated states in a time step dt in a pool of S cells.

    Arguments:
        m: number of homozygous methylated cells - array of the ints
        k: number of heterozygous methylated cells - array of the ints
        w: number of homozygous demethylated cells - array of the ints
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu: rate to transition from heterozygous to homozygous methylated
            - float >= 0
        zeta: rate to transition from heterozygous to homozygous demethylated
            - float >= 0         
        S: total number of cells all(m + k + w == S) - int
        dt: time step - float > 0
        rng: np.random.default_rng() object, Optional
    Returns:
        Updated m, k, w after transitions have occurred
    """

    if rng is None:
        rng = np.random.default_rng()

    NSIM = len(m)

    # Use sequential rounds of binomial sampling to calculate how many cells
    # transition between each state
    m_to_k, k_out, w_to_k = rng.binomial(
                                    n = (m, k, w), 
                                    p = np.tile([2*gamma*dt, 
                                        (nu + zeta)*dt, 2*mu*dt], [NSIM, 1]).T)

    k_to_m = rng.binomial(n=k_out, p = np.repeat(nu / (nu + zeta), NSIM))

    m = m - m_to_k + k_to_m
    k = k - k_out + m_to_k + w_to_k
    w = S - m - k

    return (m, k, w)

def multinomial_rvs(counts, p, rng=None):
    """
    Simulate multinomial sampling of D dimensional probability distribution

    Arguments:
        counts: number of draws from distribution - int or array of the 
                ints (N)
        p: probability  - array of the floats (D, N)
        rng: np.random.default_rng() object, Optional
    Returns:
        Multinomial sample
    """

    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(counts, (np.ndarray)):
        counts = np.full(p[0, ...].shape, counts)

    out = np.zeros(np.shape(p), dtype=int)
    ps = np.cumsum(p[::-1, ...], axis=0)[::-1, ...]
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0

    for i in range(p.shape[0]-1):
        binsample = rng.binomial(counts, condp[i, ...])
        out[i, ...] = binsample
        counts -= binsample

    out[-1, ...] = counts

    return out

def initialise_cancer(tau, mu, gamma, nu, zeta, NSIM, rng=None, init = None):
    """
    Initialise a cancer, assigning fCpG states assuming fCpGs are homozygous 
    at t=0

    Arguments:
        tau: age when population began expanding exponentially - float
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu: rate to transition from heterozygous to homozygous methylated
            - float >= 0
        zeta: rate to transition from heterozygous to homozygous demethylated
            - float >= 0         
        NSIM: number of fCpG loci to simulate - int
        rng: np.random.default_rng() object, Optional
        init: allowed values 0, 1, 2 or None. If None, initialise assuming
                time cancer began at time time tau, otherwise initailise in 
                0: w, 1: k, or 2: m. 
    Returns:
        m_cancer, k_cancer, w_cancer: number of homo meth, hetet meth and 
                homo unmeth cells in the population - np.array[int]
    """

    if rng is None:
        rng = np.random.default_rng()

    if init is None:
        # assume fCpG's are homozygous methylated at t=0
        mkw = np.zeros((3, NSIM), dtype = int)
        idx = np.arange(NSIM)
        np.random.shuffle(idx)
        mkw[0, idx[:NSIM//2]] = 1
        mkw[2, idx[NSIM//2:]] = 1

        # generate distribution of fCpG loci when population begins growing 
        # at t=tau
        RateMatrix = np.array([[-2*gamma, nu, 0], 
                                [2*gamma, -(nu+zeta), 2*mu], 
                                [0, zeta, -2*mu]])

        ProbStates = linalg.expm(RateMatrix * tau) @ mkw

        m_cancer, k_cancer, w_cancer = multinomial_rvs(1, ProbStates, rng)
    
    elif init in [0, 1, 2]:
        wkm = np.zeros((3, NSIM), dtype = int)
        wkm[init, :] = 1

        w_cancer, k_cancer, m_cancer = wkm

    else:
        raise ValueError('init must be None or 0, 1 or 2')

    return m_cancer, k_cancer, w_cancer

def grow_cancer(m_cancer, k_cancer, w_cancer, S_cancer_i, S_cancer_iPlus1, rng):
    """
    Grow a cancer, assigning fCpG states according to a multinomial ditribution

    Arguments:
        m_cancer, k_cancer, w_cancer: number of homo meth, hetet meth and 
                homo unmeth cells in the population - np.array[int]
        S_cancer_i: number of cells at time t - int = m_cancer + k_cancer + w_cancer
        S_cancer_iPlus1: number of cells at time t+dt - int >= S_cancer_i
        rng: np.random.default_rng() object, Optional
    Returns:
        Updated m_cancer, k_cancer, w_cancer
    """

    if rng is None:
        rng = np.random.default_rng()

    if S_cancer_iPlus1 - S_cancer_i > 0:
        prob_matrix = np.stack((m_cancer, k_cancer, w_cancer)) / S_cancer_i
        growth = multinomial_rvs(S_cancer_iPlus1 - S_cancer_i, prob_matrix, rng)

        m_cancer += growth[0, :]
        k_cancer += growth[1, :]
        w_cancer += growth[2, :]

    return m_cancer, k_cancer, w_cancer


def stochastic_growth(theta, tau, mu, gamma, nu, zeta, T, NSIM, init = None):
    """
    Simulate the methylation distribution of fCpG loci for an exponentially 
    growing well-mixed population evolving neutrally

    Arguments:
        theta: exponential growth rate of population - float
        tau: age when population began expanding exponentially - float < T
        mu: rate to transition from homozygous demethylated to heterozygous
            - float >= 0
        gamma: rate to transition from homozygous methylated to heterozygous
            - float >= 0
        nu: rate to transition from heterozygous to homozygous methylated
            - float >= 0
        zeta: rate to transition from heterozygous to homozygous demethylated
            - float >= 0         
        T: patient's age - float
        NSIM: number of fCpG loci to simulate - int
    Returns:
        betaCancer: fCpG methylation fraction distribution - np.array[float]
    """

    # calculate the time step so all transition probabilities are <= 10%
    dt_max = 0.01 / np.max((
        2*gamma, 
        2*mu,
        2*nu,
        2*zeta,
        theta)
    )
    
    # calculate deterministic exponential growth population size
    n = int((T-tau) / dt_max) + 2  # Number of time steps.
    t = np.linspace(tau, T, n) 
    dt = t[1] - t[0]
    S_cancer = np.exp(theta * (t-tau)).astype(int)

    if np.any(S_cancer < 0):
        raise(OverflowError('overflow encountered for S_cancer'))

    rng = np.random.default_rng()

    # generate distribution of fCpG loci depending on init param
    m_cancer, k_cancer, w_cancer = initialise_cancer(tau, mu, gamma, nu, zeta, 
                                                     NSIM, rng, init)

    # simulate changes to methylation distribution by splitting the process 
    # into 2 phases, an exponential growth phase and a methylation transition 
    # phase
    for i in range(len(t)-1):
        m_cancer, k_cancer, w_cancer = grow_cancer(m_cancer, k_cancer,
                                                    w_cancer, S_cancer[i], 
                                                    S_cancer[i+1], rng)

        m_cancer, k_cancer, w_cancer = generate_next_timepoint(m_cancer, 
                                                    k_cancer, w_cancer, 
                                                    mu, gamma, nu, zeta,
                                                    S_cancer[i+1], dt, rng)

    with np.errstate(divide='raise', over='raise'):
        betaCancer = (k_cancer + 2*m_cancer) / (2*S_cancer[-1])

    return betaCancer

def calculate_mixing_weight_lpmf(mu, gamma, nu, zeta, tau):
    # generate distribution of fCpG loci when population begins growing 
    # at t=tau
    RateMatrix = np.array([[-2*gamma, nu, 0], 
                            [2*gamma, -(nu+zeta), 2*mu], 
                            [0, zeta, -2*mu]])
    
    # assume population is either homozygous methylated or demethylated at t=0
    mkw = np.array([0.5, 0, 0.5]).T

    # use matrix exponentiation to solve to for the mixing weights
    ProbStates = linalg.expm(RateMatrix * tau) @ mkw

    return np.log(ProbStates)

def calculate_frechet_parameters(mut_rate, theta, deltaT):

    a = np.exp(1) / 2
    m_norm = 0.5 * mut_rate / theta * (theta * deltaT + np.log(mut_rate / theta ) - (1 + a))
    s_norm = 0.5 * np.exp(1) * mut_rate / theta

    return a, m_norm, s_norm

def demeth_homo_init_lpdf(y, theta, mu, tau, T):

    a, m_norm, s_norm = calculate_frechet_parameters(2 * mu, theta, T - tau)

    return invweibull.logpdf(y, a, loc = m_norm, scale = s_norm)

def meth_homo_init_lpdf(y, theta, gamma, tau, T):

    a, m_norm, s_norm = calculate_frechet_parameters(2 * gamma, theta, T - tau)

    return invweibull.logpdf(1-y, a, loc = m_norm, scale = s_norm)

def hetero_integrand(x, z, theta, nu, zeta, tau, T):

    a, m1_norm, s1_norm = calculate_frechet_parameters(nu, theta, T - tau)
    a, m2_norm, s2_norm = calculate_frechet_parameters(zeta, theta, T - tau)

    return np.exp(invweibull.logpdf(x, a, loc = m2_norm, scale = s2_norm) 
                  + invweibull.logpdf(z-0.5+x, a, loc = m1_norm, scale = s1_norm))

def meth_hetero_init_lpdf(y, theta, nu, zeta, tau, T):
    
    if isinstance(y, (list, tuple, np.ndarray)):
        integral = np.array([quad(hetero_integrand, 0, 1, 
                                args=(y_i, theta, nu, zeta, tau, T))[0]
                            for y_i in y])
    else:
        integral = quad(hetero_integrand, 0, 1, 
                            args=(y, theta, nu, zeta, tau, T))[0]

    return np.log(integral)

def combined_lpdf(y, theta, tau, mu, gamma, nu, zeta, T):

    mixing_lpdf = calculate_mixing_weight_lpmf(mu, gamma, nu, zeta, tau)

    left_peak = demeth_homo_init_lpdf(y, theta, mu, tau, T)
    central_peak = meth_hetero_init_lpdf(y, theta, nu, zeta, tau, T)
    right_peak = meth_homo_init_lpdf(y, theta, gamma, tau, T)

    # Combine using logsumexp: log(w1*p1 + w2*p2 + w3*p3) = 
    # logsumexp([log(w1)+log(p1), log(w2)+log(p2), log(w3)+log(p3)])
    log_weighted_components = np.array([
        mixing_lpdf[0] + right_peak,    # log(w1) + log(p1) for demethylated homozygous
        mixing_lpdf[1] + central_peak, # log(w2) + log(p2) for heterozygous  
        mixing_lpdf[2] + left_peak    # log(w3) + log(p3) for methylated homozygous
    ])
    
    # Use logsumexp to compute the final combined log probability density
    combined_lpdf = logsumexp(log_weighted_components, axis=0)

    return combined_lpdf

def fit_landau_rvs(beta):
    loc, scale = stats.landau.fit(beta)
    return stats.landau.rvs(loc, scale, size=10000)

def calculate_scaling_factor(beta):
    hist = np.histogram(beta, bins=np.linspace(0, 1, 201), density=True)
    nondense_hist = np.histogram(beta, bins=np.linspace(0, 1, 201), density=False)
    prop = np.max(nondense_hist[0])/np.sum(nondense_hist[0])
    scale = np.max(hist[0])/prop

    return scale

def calculate_spike_height(beta, mu):
    scale = calculate_scaling_factor(beta)

    return 2 * scale * mu * np.log(2)/theta

T = 50
tau = 45
theta = 2.4
mu = 0.07
gamma = 1e-12
nu = 1e-12
zeta = 1e-12

init = 0

betaCancer = stochastic_growth(theta, tau, mu, gamma, nu, zeta, T, 10000, init)

x = np.linspace(0, 1, 1001)
# pdf = np.exp(combined_lpdf(x, theta, tau, mu, gamma, nu, zeta, T))

fig, ax = plt.subplots()
# frechet = stats.invweibull.rvs(*calculate_frechet_parameters(2*mu, theta, T-tau), size=10000)
# plt.hist(frechet, bins = np.linspace(0, 1, 201), 
        #  alpha = 0.4, density = True)


spike = calculate_spike_height(betaCancer, mu)

plt.hist(betaCancer, bins = np.linspace(0, 1, 201), 
         alpha = 0.4, density = True)
plt.plot([0.5025, 0.5025], [0, spike], alpha=0.6, linewidth=10)
# plt.plot(x, pdf)
plt.xlabel('Population size')
plt.ylabel('Probability density')
plt.xlim(0.45, 0.55)
plt.ylim(0, 10)
plt.tight_layout()
plt.legend(labels=["Analytic Approximation", "Stochastic Distribution"])
sns.despine()
plt.show()