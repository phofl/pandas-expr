from __future__ import annotations

import functools
from collections.abc import Sequence
from types import LambdaType

from pandas_expr._deps import normalize_token, tokenize


def _convert_to_list(column) -> list | None:
    if column is None or isinstance(column, list):
        pass
    elif isinstance(column, tuple):
        column = list(column)
    elif hasattr(column, "dtype"):
        column = column.tolist()
    else:
        column = [column]
    return column


def is_scalar(x):
    return not (isinstance(x, Sequence) or hasattr(x, "dtype")) or isinstance(x, str)


@normalize_token.register(LambdaType)
def _normalize_lambda(func):
    return str(func)


def _tokenize_deterministic(*args, **kwargs):
    # Utility to be strict about deterministic tokens
    return tokenize(*args, **kwargs)


def ishashable(x):
    """Is x hashable?

    Examples
    --------

    >>> ishashable(1)
    True
    >>> ishashable([1])
    False
    """
    try:
        hash(x)
        return True
    except TypeError:
        return False


def apply(func, args, kwargs=None):
    """Apply a function given its positional and keyword arguments.

    Equivalent to ``func(*args, **kwargs)``
    Most Dask users will never need to use the ``apply`` function.
    It is typically only used by people who need to inject
    keyword argument values into a low level Dask task graph.

    Parameters
    ----------
    func : callable
        The function you want to apply.
    args : tuple
        A tuple containing all the positional arguments needed for ``func``
        (eg: ``(arg_1, arg_2, arg_3)``)
    kwargs : dict, optional
        A dictionary mapping the keyword arguments
        (eg: ``{"kwarg_1": value, "kwarg_2": value}``

    Examples
    --------
    >>> from dask.utils import apply
    >>> def add(number, second_number=5):
    ...     return number + second_number
    ...
    >>> apply(add, (10,), {"second_number": 2})  # equivalent to add(*args, **kwargs)
    12

    >>> task = apply(add, (10,), {"second_number": 2})
    >>> dsk = {'task-name': task}  # adds the task to a low level Dask task graph
    """
    if kwargs:
        return func(*args, **kwargs)
    else:
        return func(*args)


_method_cache: dict[str, methodcaller] = {}


class methodcaller:
    """
    Return a callable object that calls the given method on its operand.

    Unlike the builtin `operator.methodcaller`, instances of this class are
    cached and arguments are passed at call time instead of build time.
    """

    __slots__ = ("method",)
    method: str

    @property
    def func(self) -> str:
        # For `funcname` to work
        return self.method

    def __new__(cls, method: str):
        try:
            return _method_cache[method]
        except KeyError:
            self = object.__new__(cls)
            self.method = method
            _method_cache[method] = self
            return self

    def __call__(self, __obj, *args, **kwargs):
        return getattr(__obj, self.method)(*args, **kwargs)

    def __reduce__(self):
        return (methodcaller, (self.method,))

    def __str__(self):
        return f"<{self.__class__.__name__}: {self.method}>"

    __repr__ = __str__


class MethodCache:
    """Attribute access on this object returns a methodcaller for that
    attribute.

    Examples
    --------
    >>> a = [1, 3, 3]
    >>> M.count(a, 3) == a.count(3)
    True
    """

    def __getattr__(self, item):
        return methodcaller(item)

    def __dir__(self):
        return list(_method_cache)


M = MethodCache()


def funcname(func) -> str:
    """Get the name of a function."""
    # functools.partial
    if isinstance(func, functools.partial):
        return funcname(func.func)
    # methodcaller
    if isinstance(func, methodcaller):
        return func.method[:50]

    module_name = getattr(func, "__module__", None) or ""
    type_name = getattr(type(func), "__name__", None) or ""

    # toolz.curry
    if "toolz" in module_name and "curry" == type_name:
        return func.func_name[:50]

    # All other callables
    try:
        name = func.__name__
        if name == "<lambda>":
            return "lambda"
        return name[:50]
    except AttributeError:
        return str(func)[:50]
