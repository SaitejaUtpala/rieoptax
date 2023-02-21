from jax import tree_util


def apply_exp_updates(params, updates, manifold_dict):
    new_params = tree_util.tree_map(
        lambda param, update, manifold: manifold.exp(param, update),
        params,
        updates,
        manifold_dict,
    )
    return new_params


def apply_retr_updates(params, updates, manifold_dict):
    new_params = tree_util.tree_map(
        lambda param, update, manifold: manifold.retr(param, update),
        params,
        updates,
        manifold_dict,
    )
    return new_params


def apply_updates(params, updates, manifold_dict, update_fn="exp"):
    assert update_fn in ["exp", "retr"]

    return (
        apply_exp_updates(params=params, updates=updates, manifold_dict=manifold_dict)
        if update_fn is "exp"
        else apply_retr_updates(
            params=params, updates=updates, manifold_dict=manifold_dict
        )
    )
