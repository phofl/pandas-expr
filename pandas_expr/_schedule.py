import sys

from dask.core import flatten, get_dependencies
from dask.local import finish_task, start_state_from_dask
from dask.order import order

from pandas_expr._util import ishashable


def identity(x):
    """Identity function. Returns x.

    >>> identity(3)
    3
    """
    return x


def pack_exception(e, dumps):
    return e, sys.exc_info()[2]


def istask(x):
    """Is x a runnable task?

    A task is a tuple with a callable first argument

    Examples
    --------

    >>> inc = lambda x: x + 1
    >>> istask((inc, 1))
    True
    >>> istask(1)
    False
    """
    return type(x) is tuple and x and callable(x[0])


def _execute_task(arg, cache, dsk=None):
    """Do the actual work of collecting data and executing a function

    Examples
    --------

    >>> inc = lambda x: x + 1
    >>> add = lambda x, y: x + y
    >>> cache = {'x': 1, 'y': 2}

    Compute tasks against a cache
    >>> _execute_task((add, 'x', 1), cache)  # Compute task in naive manner
    2
    >>> _execute_task((add, (inc, 'x'), 1), cache)  # Support nested computation
    3

    Also grab data from cache
    >>> _execute_task('x', cache)
    1

    Support nested lists
    >>> list(_execute_task(['x', 'y'], cache))
    [1, 2]

    >>> list(map(list, _execute_task([['x', 'y'], ['y', 'x']], cache)))
    [[1, 2], [2, 1]]

    >>> _execute_task('foo', cache)  # Passes through on non-keys
    'foo'
    """
    if isinstance(arg, list):
        return [_execute_task(a, cache) for a in arg]
    elif istask(arg):
        func, args = arg[0], arg[1:]
        # Note: Don't assign the subtask results to a variable. numpy detects
        # temporaries by their reference count and can execute certain
        # operations in-place.
        return func(*(_execute_task(a, cache) for a in args))
    elif not ishashable(arg):
        return arg
    elif arg in cache:
        return cache[arg]
    else:
        return arg


def schedule(dsk, result, cache=None, **kwargs):
    if isinstance(result, list):
        result_flat = set(flatten(result))
    else:
        result_flat = {result}
    results = set(result_flat)

    dsk = dict(dsk)

    keyorder = order(dsk)

    state = start_state_from_dask(dsk, cache=cache, sortkey=keyorder.get)

    if state["waiting"] and not state["ready"]:
        raise ValueError("Found no accessible jobs")

    # Main loop, wait on tasks to finish, insert new ones
    while state["waiting"] or state["ready"] or state["running"]:
        key = state["ready"].pop()
        # Notify task is running
        state["running"].add(key)

        data = {dep: state["cache"][dep] for dep in get_dependencies(dsk, key)}

        result = _execute_task(dsk[key], data)
        state["cache"][key] = result
        finish_task(dsk, key, state, results, keyorder.get)

    return result
