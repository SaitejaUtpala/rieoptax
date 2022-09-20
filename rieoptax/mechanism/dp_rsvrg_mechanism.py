

from math import sqrt

from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import AmplificationBySampling, Composition
from scipy.optimize import minimize_scalar


class DP_RSVRG_Mechanism(Mechanism):
    def __init__(self, params, name="DP-RiemannianSVRG"):
        Mechanism.__init__(self)
        self.name = name
        self.params = params
        subsample = AmplificationBySampling(PoissonSampling=False)
        full_gradient_mechanism = GaussianMechanism(sigma=params["sigma1"])
        stochastic_gradient_mechanism = GaussianMechanism(sigma=params["sigma2"])
        stochastic_gradient_mechanism.neighboring = "replace_one"
        subsampled_stochastic_mechanism = subsample(
            stochastic_gradient_mechanism, params["prob"], improved_bound_flag=True
        )
        compose = Composition()
        mech = compose(
            [full_gradient_mechanism, subsampled_stochastic_mechanism],
            [params["epochs"], params["epochs"] * params["frequency"]],
        )
        self.set_all_representation(mech)

    @staticmethod
    def get_eps(sigma, delta, n, epochs, frequency, L1, L2):
        def f(a, sigma):
            svrg_params = {}
            svrg_params["sigma1"] = sigma * sqrt(a) * (n / (2 * L1))
            svrg_params["sigma2"] = sigma * sqrt((1 - a)) * (1 / (4 * L2))
            svrg_params["epochs"] = epochs
            svrg_params["frequency"] = frequency
            svrg_params["prob"] = 1 / n
            dp_svrg = DP_RSVRG_Mechanism(svrg_params)
            eps = dp_svrg.get_approxDP(delta=delta)
            return eps

        res = minimize_scalar(lambda a: f(a, sigma), bounds=(0, 1), method="bounded")
        curr_eps = f(res.x, sigma)
        return curr_eps

    @staticmethod
    def get_sigma(params):
        print("Searching for best sigma...")

        def err(sigma, eps, delta, n, epochs, frequency, L1, L2):
            return abs(
                eps - DP_RSVRG_Mechanism.get_eps(sigma, delta, n, epochs, frequency, L1, L2)
            )

        results = minimize_scalar(
            lambda sigma: err(
                sigma,
                params["eps"],
                params["delta"],
                params["n"],
                params["epochs"],
                params["frequency"],
                params["L1"],
                params["L2"],
            ),
            method="bounded",
            bounds=(0, 10),
            options= {'xatol': 1e-08, 'maxiter': 60000}
        )
        if results.success and results.fun < 2*1e-3:
            print("Found!")
            print("re", results.fun)
        else:
            print("re", results.fun)
            raise RuntimeError(
                f"eps_delta_calibrator fails to find a parameter: {results.message}"
            )
        return results.x