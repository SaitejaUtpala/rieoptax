from rieoptax.core import RiemannianGradientTransformation


def chain(*args):
    init_fns, update_fns = zip(*args)

    def init_fn(params):
        return tuple(fn(params) for fn in init_fn)

    def update_fn(updates, state, params=None):
        if len(update_fns) != len(state):
            raise ValueError('The number of updates and states has to be the same in'
            'chain! Make sure you have called init first')

        new_state = []
        for s, fn in zip(state, update_fns):
            updates, new_s = fn(updates, s, params)
            new_state.append(new_s)

        return updates, tuple(new_state)

    return RiemannianGradientTransformation(init_fn, update_fn)





