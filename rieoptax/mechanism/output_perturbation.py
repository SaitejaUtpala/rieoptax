
from math import exp, log, sqrt

SQRT_2 = sqrt(2)

from math import exp, sqrt

from jax import numpy as jnp
from scipy.special import erf


def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):


    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma

class LogEuclideanSampleronSPD():

    def vecd(self, spd):
        """vectorization operator

        Parameters
        ----------
        spd : np array , shape [k, k]
            SPD matrix.

        Returns
        -------
        vec : np array, shape [k*(k+1)/2]
        """
        k, k = spd.shape

        upper_triangular=  SQRT_2* spd[np.triu_indices(k, 1)]
        diag = np.diag(spd)
        return np.hstack([diag, upper_triangular])

    def inv_vecd(self, vector):
        """inverse vectorization operator

        Parameters
        ----------
        spd : vector, [k*(k+1)/2]
            SPD matrix.

        Returns
        -------
        vec : np array, shape [k]
        """

        dim = vector.shape[0]
        k = (int)((sqrt(8*dim +1) - 1)/2)
        diag_indices = np.diag_indices(k)
        triu_indices = np.triu_indices(k,1)
        spd = np.zeros((k,k))
        spd[diag_indices] = vector[:k]/2
        spd[triu_indices] = vector[k:]/SQRT_2
        spd = spd + spd.T
        return spd

    def logm(self, spd):
        """Matrix logarithm"""
        eigval, eigvec = np.linalg.eigh(spd)
        return eigvec @ np.diag(np.log(eigval)) @ eigvec.T

    def expm(self, spd):
        """Matrix exponential"""
        eigval, eigvec = np.linalg.eigh(spd)
        return eigvec @ np.diag(np.exp(eigval)) @ eigvec.T

    def sample(self, mean, std): 
        vecd_log_mean = self.vecd(self.logm(mean))
        var = np.array([std]*vecd_log_mean.shape[0])
        normal = np.random.normal(vecd_log_mean, var)
        sample = self.expm(self.inv_vecd(normal))
        return sample



class LogEuclideanGaussianMechanismOnSPD():

    def __init__(self, k, sigma_type='classical') -> None:
        self.k = k 
        self.sigma_type = sigma_type
        self.sampler = LogEuclideanSampleronSPD()
        #self.metric = SPDMetricLogEuclidean(k)

    def classical_sigma(self, sensitivity, epsilon, delta):
        """calculate sigma based on classical gaussian mechanism.

        Parameters
        ----------
        sensitivity : real
                Log Euclidean sensitivity.
        epsilon : real
                Epsilon in (epsilon, delta)-DP.
        delta : real
                Delta in (epsilon, delta)-DP.

        Returns
        -------
            _ : real.
                Standard Deviation. 
        """
        sigma = (sensitivity/epsilon)**2 * 2 * log(1.25/delta)
        return sqrt(sigma)

    def calibrated_analytic_sigma(self, sensitivity, epsilon, delta):
        """calculate sigma based on calibrated analytic mechanism.

        Parameters
        ----------
        sensitivity : real
                Log Euclidean sensitivity.
        epsilon : real
                Epsilon in (epsilon, delta)-DP.
        delta : real
                Delta in (epsilon, delta)-DP.

        Returns
        -------
            _ : real.
                Standard Deviation. 
        """
        return calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)


    def privatize(self, f, sensitivity, epsilon, delta):
        """Private f"""
       
        if self.sigma_type == 'classical':
            sigma = self.classical_sigma(sensitivity, epsilon, delta)
        elif self.sigma_type == 'calibrated_analytic':
           
            sigma = self.calibrated_analytic_sigma(sensitivity, epsilon, delta)

        #print("sigma", sigma)
        private_f = self.sampler.sample(f, sigma)
        dist = self.metric.dist(private_f, f)
        return private_f, dist