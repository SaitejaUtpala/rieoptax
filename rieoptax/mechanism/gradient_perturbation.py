from autodp.autodp_core import Mechanism
from autodp.mechanism_zoo import GaussianMechanism
from autodp.transformer_zoo import AmplificationBySampling, Composition
from scipy.optimize import minimize_scalar


class DP_RGD_Mechanism(Mechanism):
    def __init__(self, params, name="DP-GD"):
        Mechanism.__init__(self)
        self.name = name
        self.params = params

        full_gradient_mechanism = GaussianMechanism(sigma=params["sigma"])
        compose = Composition()
        mech = compose([full_gradient_mechanism], [params["epochs"]])
        self.set_all_representation(mech)

    @staticmethod
    def get_eps(sigma, delta, n, epochs, L):
        gd_params = {}
        gd_params["sigma"] = sigma * (n / (2 * L))
        gd_params["epochs"] = epochs
        dp_gd = DP_RGD_Mechanism(gd_params)
        eps = dp_gd.get_approxDP(delta=delta)
        #print("eps", eps)
        return eps

    @staticmethod
    def get_sigma(params):
        def err(sigma, eps, delta, n, epochs, L):
            return abs(eps - DP_RGD_Mechanism.get_eps(sigma, delta, n, epochs, L))

        results = minimize_scalar(
            lambda sigma: err(sigma, params["eps"], params["delta"], params["n"], params["epochs"], params["L"]),
            method="bounded",
            bounds=(0, 20),
            options= {'xatol': 1e-06, 'maxiter': 70000}
        )
        if results.success and results.fun < 5*1e-3:
            print("Worked!")
            print("s", results.fun)
        else:
            print("s", results.success)
            print("s", results.fun)
            raise RuntimeError(
                f"eps_delta_calibrator fails to find a parameter: {results.message}"
            )
        return results.x

class DP_RSGD_Mechanism(Mechanism):
    def __init__(self, params, name="DP-RSGD"):
        Mechanism.__init__(self)
        self.name = name
        self.params = params
        subsample = AmplificationBySampling(PoissonSampling=False)
        stochastic_gradient_mechanism = GaussianMechanism(sigma=params["sigma"])
        stochastic_gradient_mechanism.neighboring = "replace_one"
        subsampled_stochastic_mechanism = subsample(
            stochastic_gradient_mechanism, params["prob"], improved_bound_flag=True
        )
        compose = Composition()
        mech = compose([subsampled_stochastic_mechanism], [params["epochs"]])
        self.set_all_representation(mech)

    @staticmethod
    def get_eps(sigma, delta, n, epochs, L):
        sgd_params = {}
        sgd_params["sigma"] = sigma * (1 / (2 * L))
        sgd_params["epochs"] = epochs
        sgd_params["prob"] = 1 / n
        dp_sgd = DP_RSGD_Mechanism(sgd_params)
        eps = dp_sgd.get_approxDP(delta=delta)
        return eps

    @staticmethod
    def get_sigma(params):
        def err(sigma, eps, delta, n, epochs, L):
            return abs(eps - DP_RSGD_Mechanism.get_eps(sigma, delta, n, epochs, L))

        results = minimize_scalar(
            lambda sigma: err(
                sigma,
                params["eps"],
                params["delta"],
                params["n"],
                params["epochs"],
                params["L"],
            ),
            method="bounded",
            bounds=(0, 10),
            options= {'xatol': 1e-08, 'maxiter': 60000}
        )
        if results.success and results.fun < 2*1e-3:
            print("Worked!")
            print("s", results.fun)
        else:
            raise RuntimeError(
                f"eps_delta_calibrator fails to find a parameter: {results.message}"
            )
        return results.x