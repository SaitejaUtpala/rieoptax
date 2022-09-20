
def RGD(problem, params):
    loss_list = []
    gradient_complexity = []

    w_curr = problem.init
    loss_list.append(problem.objective_value(w_curr))
    gradient_complexity.append(0)

    if params["private"]:
        problem.metric.gen_and_cache_samples(params["sigma"], params["sampling_calls"])
    for i in tqdm(range(params["epochs"])):
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
    return np.array(loss_list), np.cumsum(gradient_complexity)