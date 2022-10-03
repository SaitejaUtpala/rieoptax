import core


def chain(*args):
    """Applies a list of chainable update transformations

    Given a sequnce of chainable transforms, `chain` returns 
    an `init_fn` by concatenating the states of the individual 
    transform, and returns an `update_fn` feeding the appropriate
    state to each other.

    Args:
        *args : a sequence of chainable(init_fn, update_fn) tuples.

    Returns:
        A single (init_fn, update_fn) tuple.
    """

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

    return core.RiemannainGradientTransformation(init_fn, update_fn)





