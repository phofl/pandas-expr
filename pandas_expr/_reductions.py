import pandas as pd
from dask.dataframe.core import make_meta, meta_nonempty

from pandas_expr._expr import Elemwise, Expr, Index, Projection
from pandas_expr._util import M, apply


class Reduction(Expr):
    _parameters = ["frame"]
    reducer = None
    reduction_kwargs = {}

    def _layer(self):
        # Normalize functions in case not all are defined
        reducer = self.reducer
        reduction_kwargs = self.reduction_kwargs

        d = {}
        keys = self.frame.__dask_keys__()

        # apply reducer to every input
        for key in keys:
            if reduction_kwargs:
                d[self._name] = (apply, reducer, [key], reduction_kwargs)
            else:
                d[self._name] = (reducer, key)

        return d

    @property
    def _meta(self):
        meta = meta_nonempty(self.frame._meta)
        return self.reducer(meta, **self.reduction_kwargs)

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            return type(self)(self.frame[parent.operand("columns")], *self.operands[1:])


class Unique(Reduction):
    _parameters = ["frame"]
    reducer = M.unique

    @property
    def _meta(self):
        return self.reducer(meta_nonempty(self.frame._meta))

    def _simplify_up(self, parent):
        return


class DropDuplicates(Unique):
    _parameters = ["frame", "subset", "ignore_index"]
    _defaults = {"subset": None, "ignore_index": False}
    reducer = M.drop_duplicates

    @property
    def _meta(self):
        return self.reducer(meta_nonempty(self.frame._meta), **self.reduction_kwargs)

    def _subset_kwargs(self):
        if isinstance(self.frame._meta, pd.Series):
            return {}
        return {"subset": self.subset}

    @property
    def reduction_kwargs(self):
        return {"ignore_index": self.ignore_index, **self._subset_kwargs()}

    def _simplify_up(self, parent):
        if self.subset is not None:
            columns = set(parent.columns).union(self.subset)
            if columns == set(self.frame.columns):
                # Don't add unnecessary Projections, protects against loops
                return

            return type(parent)(
                type(self)(self.frame[sorted(columns)], *self.operands[1:]),
                *parent.operands[1:],
            )


class Sum(Reduction):
    _parameters = ["frame", "skipna", "numeric_only", "min_count"]
    reducer = M.sum

    @property
    def reduction_kwargs(self):
        return dict(
            skipna=self.skipna,
            numeric_only=self.numeric_only,
            min_count=self.min_count,
        )


class Prod(Reduction):
    _parameters = ["frame", "skipna", "numeric_only", "min_count"]
    reducer = M.prod

    @property
    def reduction_kwargs(self):
        return dict(
            skipna=self.skipna,
            numeric_only=self.numeric_only,
            min_count=self.min_count,
        )


class Max(Reduction):
    _parameters = ["frame", "skipna"]
    reducer = M.max

    @property
    def reduction_kwargs(self):
        return dict(
            skipna=self.skipna,
        )


class Any(Reduction):
    _parameters = ["frame", "skipna"]
    reducer = M.any

    @property
    def reduction_kwargs(self):
        return dict(
            skipna=self.skipna,
        )


class All(Reduction):
    _parameters = ["frame", "skipna"]
    reducer = M.all

    @property
    def reduction_kwargs(self):
        return dict(
            skipna=self.skipna,
        )


class IdxMin(Reduction):
    _parameters = ["frame", "skipna"]
    reducer = M.idxmin

    @property
    def reduction_kwargs(self):
        return dict(skipna=self.skipna)


class IdxMax(IdxMin):
    reducer = M.idxmax


class Len(Reduction):
    reducer = staticmethod(len)

    def _simplify_down(self):
        from pandas_expr.io.io import IO

        # We introduce Index nodes sometimes.  We special case around them.
        if isinstance(self.frame, Index) and isinstance(self.frame.frame, Elemwise):
            return Len(self.frame.frame)

        # Let the child handle it.  They often know best
        if isinstance(self.frame, IO):
            return self

        # Drop all of the columns, just pass through the index
        if len(self.frame.columns):
            return Len(self.frame.index)

    def _simplify_up(self, parent):
        return


class Size(Reduction):
    reducer = staticmethod(lambda df: df.size)

    def _simplify_down(self):
        from pandas_expr._collection import DataFrame

        if isinstance(self.frame, DataFrame) and len(self.frame.columns) > 1:
            return len(self.frame.columns) * Len(self.frame)
        else:
            return Len(self.frame)

    def _simplify_up(self, parent):
        return


class NBytes(Reduction):
    # Only supported for Series objects
    reducer = lambda _, ser: ser.nbytes


class Var(Reduction):
    _parameters = ["frame", "skipna", "ddof", "numeric_only"]
    _defaults = {"skipna": True, "ddof": 1, "numeric_only": False}
    reducer = M.var

    @property
    def _meta(self):
        return make_meta(
            self.reducer(
                meta_nonempty(self.frame._meta),
                skipna=self.skipna,
                numeric_only=self.numeric_only,
            )
        )

    @property
    def reduction_kwargs(self):
        return dict(skipna=self.skipna, numeric_only=self.numeric_only, ddof=self.ddof)


class Std(Var):
    reducer = M.std


class Mean(Reduction):
    _parameters = ["frame", "skipna", "numeric_only"]
    _defaults = {"skipna": True, "numeric_only": False}
    reducer = M.mean

    @property
    def _meta(self):
        return (
            self.frame._meta.sum(skipna=self.skipna, numeric_only=self.numeric_only) / 2
        )


class Count(Reduction):
    _parameters = ["frame", "numeric_only"]
    reducer = M.count


class Min(Max):
    reducer = M.min


class Mode(Reduction):
    """

    Mode was a bit more complicated than class reductions, so we retreat back
    to ApplyConcatApply
    """

    _parameters = ["frame", "dropna"]
    _defaults = {"dropna": True}
    reducer = M.mode

    @property
    def reduction_kwargs(self):
        return {"dropna": self.dropna}


class ReductionConstantDim(Reduction):
    """
    Some reductions reduce the number of rows in your object but keep the original
    dimension, e.g. a DataFrame stays a DataFrame instead of getting reduced to
    a Series.
    """

    pass


class NLargest(ReductionConstantDim):
    _defaults = {"n": 5, "_columns": None}
    _parameters = ["frame", "n", "_columns"]
    reducer = M.nlargest

    def _columns_kwarg(self):
        if self._columns is None:
            return {}
        return {"columns": self._columns}

    @property
    def reduction_kwargs(self):
        return {"n": self.n, **self._columns_kwarg()}


class NSmallest(NLargest):
    _parameters = ["frame", "n", "_columns"]
    reducer = M.nsmallest


class ValueCounts(ReductionConstantDim):
    _defaults = {
        "sort": None,
        "ascending": False,
        "dropna": True,
        "normalize": False,
    }

    _parameters = ["frame", "sort", "ascending", "dropna", "normalize"]
    reducer = M.value_counts

    @property
    def reduction_kwargs(self):
        return {"sort": self.sort, "ascending": self.ascending, "dropna": self.dropna}

    def _simplify_up(self, parent):
        # We are already a Series
        return


class MemoryUsage(Reduction):
    reducer = M.memory_usage


class MemoryUsageIndex(MemoryUsage):
    _parameters = ["frame", "deep"]
    _defaults = {"deep": False}

    @property
    def reduction_kwargs(self):
        return {"deep": self.deep}


class MemoryUsageFrame(MemoryUsage):
    _parameters = ["frame", "deep", "_index"]
    _defaults = {"deep": False, "_index": True}

    @property
    def reduction_kwargs(self):
        return {"deep": self.deep, "index": self._index}
