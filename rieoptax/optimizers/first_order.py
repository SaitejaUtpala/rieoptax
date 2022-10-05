from jax import numpy as jnp




def RGD(problem, params):
    loss_list = []
    gradient_complexity = []

    w_curr = problem.init
    loss_list.append(problem.objective_value(w_curr))
    gradient_complexity.append(0)

    if params["private"]:
        problem.metric.gen_and_cache_samples(params["sigma"], params["sampling_calls"])
    for i in range(params["epochs"]):
        grad = problem.gradient(problem.Z, w_curr, params["private"],params["L"])
        gradient_complexity.append(problem.n)

        if params["private"]:  
            noise = problem.metric.sample_tangent_gaussian_zero_mean(
                w_curr, params["sigma"]
            )
            grad = grad + noise

        w_curr = problem.metric.exp(-1 * params["lr"] * grad, w_curr)
        loss = problem.objective_value(w_curr)
        loss_list.append(loss)
    return jnp.array(loss_list), jnp.cumsum(gradient_complexity)

def DP_RSGD(problem, params):

    loss_list = []
    gradient_complexity = []

    w_curr = problem.init
    loss_list.append(problem.objective_value(w_curr))
    gradient_complexity.append(0)

    if params["private"]:
        problem.metric.gen_and_cache_samples(params["sigma"], params["sampling_calls"])
    for i in tqdm(range(params["epochs"])):

        z = problem.data_sample()
        assert z.shape[0] == 1
        grad = problem.gradient(z, w_curr, params["private"],params["L"])
        gradient_complexity.append(1)

        if params["private"]:  # privitization
            noise = problem.metric.sample_tangent_gaussian_zero_mean(
                w_curr, params["sigma"]
            )
            grad = grad + noise

        w_curr = problem.metric.exp(-1 * params["lr"] * grad, w_curr)
        if (i + 1) % params["every"] == 0:
            loss = problem.objective_value(w_curr)
            loss_list.append(loss)
    return np.array(loss_list), np.cumsum(gradient_complexity)[:: params["every"]]



def DP_RSVRG(problem, params):
    loss_list = []
    gradient_complexity = []

    w_curr = problem.init
    loss_list.append(problem.objective_value(w_curr))
    gradient_complexity.append(0)


    if params["private"]:
        problem.metric.gen_and_cache_samples(params["sigma"], params["sampling_calls"])
    for i in tqdm(range(params["epochs"])):
        full_gradient = problem.gradient(problem.Z, w_curr, params["private"],params["L1"])
        w_inner = w_curr
        for j in range(params["frequency"]):
            z = problem.data_sample()
            grad1 = problem.gradient(z, w_inner, params["private"],params["L2"])
            grad2 = problem.gradient(z, w_curr, params["private"],params["L2"])

            

            grad = grad1 - problem.metric.parallel_transport(
                tangent_vec=grad2 - full_gradient, base_point=w_curr, end_point=w_inner
            )
            gradient_complexity.append(problem.n + 2 if j == 0 else 2)

            if params["private"]:  
                noise = problem.metric.sample_tangent_gaussian_zero_mean(
                    w_inner, params["sigma"]
                )
                grad = grad + noise

            w_inner = problem.metric.exp(-1 * params["lr"] * grad, w_inner)
            if (j + 1) % params["every"] == 0:
                loss = problem.objective_value(w_inner)
                loss_list.append(loss)
        w_curr = w_inner
    return jnp.array(loss_list), jnp.cumsum(gradient_complexity)[:: params["every"]]