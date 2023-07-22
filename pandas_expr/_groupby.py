import functools

import numpy as np
import pandas as pd
from dask.dataframe.dispatch import meta_nonempty

from pandas_expr._collection import DataFrame, Series, new_collection
from pandas_expr._expr import Projection
from pandas_expr._reductions import Reduction
from pandas_expr._util import M, apply


def _as_dict(key, value):
    # Utility to convert a single kwarg to a dict.
    # The dict will be empty if the value is None
    return {} if value is None else {key: value}


###
### Groupby-aggregation expressions
###


class SingleAggregation(Reduction):
    """Single groupby aggregation

    This is an abstract class. Sub-classes must implement
    the following methods:

    -   `groupby_chunk`: Applied to each group within
        the `chunk` method of `ApplyConcatApply`
    -   `groupby_aggregate`: Applied to each group within
        the `aggregate` method of `ApplyConcatApply`

    Parameters
    ----------
    frame: Expr
        Dataframe- or series-like expression to group.
    by: str, list or Series
        The key for grouping
    observed:
        Passed through to dataframe backend.
    dropna:
        Whether rows with NA values should be dropped.
    chunk_kwargs:
        Key-word arguments to pass to `groupby_chunk`.
    aggregate_kwargs:
        Key-word arguments to pass to `aggregate_chunk`.
    """

    _parameters = [
        "frame",
        "by",
        "observed",
        "dropna",
        "as_index",
        "_slice",
        "kwargs",
    ]
    _defaults = {
        "observed": None,
        "dropna": None,
        "as_index": True,
        "_slice": None,
        "kwargs": {},
    }
    reducer_func = None

    def _layer(self):
        # Normalize functions in case not all are defined
        reducer = self.reducer
        reduction_kwargs = self.reduction_kwargs

        d = {}
        keys = self.frame.__dask_keys__()

        # apply reducer to every input
        for key in keys:
            if reduction_kwargs:
                d[self._name] = (
                    apply,
                    reducer,
                    [key, self.by, self.observed, self.dropna, self.as_index],
                    reduction_kwargs,
                )
            else:
                d[self._name] = (
                    apply,
                    reducer,
                    [key, self.by, self.observed, self.dropna, self.as_index],
                )

        return d

    @property
    def kwargs(self):
        return self.operand("kwargs")

    @functools.cached_property
    def _meta(self):
        meta = meta_nonempty(self.frame._meta)
        return self.reducer(
            meta,
            self.by,
            self.observed,
            self.dropna,
            self.as_index,
            self._slice,
        )

    def reducer(
        self,
        df,
        by=None,
        observed=False,
        dropna=True,
        as_index=False,
        _slice=None,
    ):
        g = df.groupby(by, as_index=as_index, observed=observed, dropna=dropna)
        if self._slice is not None:
            g = g.__getitem__(self._slice)
        return self.reducer_func(g, **self.kwargs)

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            columns = sorted(set(parent.columns + self.by))
            if columns == self.frame.columns:
                return
            return type(parent)(
                type(self)(self.frame[columns], *self.operands[1:]),
                *parent.operands[1:],
            )


class GroupbyAggregation(SingleAggregation):
    """General groupby aggregation

    This class can be used directly to perform a general
    groupby aggregation by passing in a `str`, `list` or
    `dict`-based specification using the `arg` operand.

    Parameters
    ----------
    frame: Expr
        Dataframe- or series-like expression to group.
    by: str, list or Series
        The key for grouping
    arg: str, list or dict
        Aggregation spec defining the specific aggregations
        to perform.
    observed:
        Passed through to dataframe backend.
    dropna:
        Whether rows with NA values should be dropped.
    chunk_kwargs:
        Key-word arguments to pass to `groupby_chunk`.
    aggregate_kwargs:
        Key-word arguments to pass to `aggregate_chunk`.
    """

    _parameters = [
        "frame",
        "by",
        "arg",
        "observed",
        "dropna",
        "as_index",
        "_slice",
        "kwargs",
    ]
    reducer_func = M.agg

    @property
    def kwargs(self):
        kwargs = self.operand("kwargs")
        if kwargs is None:
            kwargs = {}
        kwargs.update({"func": self.arg})
        return kwargs

    def _simplify_down(self):
        # Use agg-spec information to add column projection
        column_projection = None
        if isinstance(self.arg, dict):
            column_projection = (
                set(self.by).union(self.arg.keys()).intersection(self.frame.columns)
            )
        if column_projection and column_projection < set(self.frame.columns):
            return type(self)(self.frame[list(column_projection)], *self.operands[1:])


class Sum(SingleAggregation):
    reducer_func = M.sum


class Prod(SingleAggregation):
    reducer_func = M.prod


class Min(SingleAggregation):
    reducer_func = M.min


class Max(SingleAggregation):
    reducer_func = M.max


class First(SingleAggregation):
    reducer_func = M.first


class Last(SingleAggregation):
    reducer_func = M.last


class Count(SingleAggregation):
    reducer_func = M.count


class Size(SingleAggregation):
    reducer_func = M.size


class ValueCounts(SingleAggregation):
    reducer_func = M.value_counts


class Var(SingleAggregation):
    reducer_func = M.var


class Std(SingleAggregation):
    reducer_func = M.std


class Mean(SingleAggregation):
    reducer_func = M.mean


###
### Groupby Collection API
###


class GroupBy:
    """Collection container for groupby aggregations

    The purpose of this class is to expose an API similar
    to Pandas' `Groupby` for dask-expr collections.

    See Also
    --------
    SingleAggregation
    """

    def __init__(
        self,
        obj,
        by,
        sort=None,
        observed=None,
        dropna=None,
        slice=None,
    ):
        if (
            isinstance(by, Series)
            and by.name in obj.columns
            and by._name == obj[by.name]._name
        ):
            by = by.name
        elif isinstance(by, Series):
            # TODO: Implement this
            raise ValueError("by must be in the DataFrames columns.")

        by_ = by if isinstance(by, (tuple, list)) else [by]
        self._slice = slice
        # Check if we can project columns
        projection = None
        if (
            np.isscalar(slice)
            or isinstance(slice, (str, list, tuple))
            or (isinstance(slice, pd.Index) or isinstance(slice, Series))
        ):
            projection = set(by_).union(
                {slice} if (np.isscalar(slice) or isinstance(slice, str)) else slice
            )
            projection = [c for c in obj.columns if c in projection]

        self.by = [by] if np.isscalar(by) else list(by)
        self.obj = obj[projection] if projection is not None else obj
        self.sort = sort
        self.observed = observed
        self.dropna = dropna

        if not isinstance(self.obj, DataFrame):
            raise NotImplementedError(
                "groupby only supports DataFrame collections for now."
            )

        for key in self.by:
            if not (np.isscalar(key) and key in self.obj.columns):
                raise NotImplementedError("Can only group on column names (for now).")

        if self.sort:
            raise NotImplementedError("sort=True not yet supported.")

    def _single_agg(self, expr_cls, **kwargs):
        return new_collection(
            expr_cls(
                self.obj.expr,
                self.by,
                self.observed,
                self.dropna,
                kwargs=kwargs,
                _slice=self._slice,
            )
        )

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(e) from e

    def __getitem__(self, key):
        g = GroupBy(
            self.obj,
            by=self.by,
            slice=key,
            sort=self.sort,
            dropna=self.dropna,
            observed=self.observed,
        )
        return g

    def count(self, **kwargs):
        return self._single_agg(Count, **kwargs)

    def sum(self, **kwargs):
        return self._single_agg(Sum, **kwargs)

    def prod(self, **kwargs):
        return self._single_agg(Prod, **kwargs)

    def mean(self, **kwargs):
        return self._single_agg(Mean, **kwargs)

    def min(self, **kwargs):
        return self._single_agg(Min, **kwargs)

    def max(self, **kwargs):
        return self._single_agg(Max, **kwargs)

    def first(self, **kwargs):
        return self._single_agg(First, **kwargs)

    def last(self, **kwargs):
        return self._single_agg(Last, **kwargs)

    def size(self, **kwargs):
        return self._single_agg(Size, **kwargs)

    def value_counts(self, **kwargs):
        return self._single_agg(ValueCounts, **kwargs)

    def var(self, ddof=1, numeric_only=False):
        return self._single_agg(Var, ddof=ddof, numeric_only=numeric_only)

    def std(self, ddof=1, numeric_only=False):
        return self._single_agg(Std, ddof=ddof, numeric_only=numeric_only)

    def aggregate(self, arg=None):
        if arg is None:
            raise NotImplementedError("arg=None not supported")

        return new_collection(
            GroupbyAggregation(
                self.obj.expr,
                self.by,
                arg,
                self.observed,
                self.dropna,
            )
        )

    def agg(self, *args, **kwargs):
        return self.aggregate(*args, **kwargs)
