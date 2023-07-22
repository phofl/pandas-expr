from __future__ import annotations

import functools

import pandas as pd
from dask.base import collections_to_dsk
from fsspec.utils import stringify_path
from pandas._typing import Axes, Dtype
from pandas.core.accessor import CachedAccessor

from pandas_expr import _expr as expr
from pandas_expr._categorical import CategoricalAccessor
from pandas_expr._concat import Concat
from pandas_expr._expr import Eval, no_default
from pandas_expr._merge import JoinRecursive, Merge
from pandas_expr._reductions import (
    DropDuplicates,
    Len,
    MemoryUsageFrame,
    MemoryUsageIndex,
    NLargest,
    NSmallest,
    Unique,
    ValueCounts,
)
from pandas_expr._schedule import schedule
from pandas_expr._util import _convert_to_list

#
# Utilities to wrap Expr API
# (Helps limit boiler-plate code in collection APIs)
#


def _wrap_expr_api(*args, wrap_api=None, **kwargs):
    # Use Expr API, but convert to/from Expr objects
    assert wrap_api is not None
    result = wrap_api(
        *[arg.expr if isinstance(arg, FrameBase) else arg for arg in args],
        **kwargs,
    )
    if isinstance(result, expr.Expr):
        return new_collection(result)
    return result


def _wrap_expr_op(self, other, op=None):
    # Wrap expr operator
    assert op is not None
    if isinstance(other, FrameBase):
        other = other.expr
    return new_collection(getattr(self.expr, op)(other))


def _wrap_unary_expr_op(self, op=None):
    # Wrap expr operator
    assert op is not None
    return new_collection(getattr(self.expr, op)())


#
# Collection classes
#


class FrameBase:
    """Base class for Expr-backed Collections"""

    __dask_optimize__ = staticmethod(lambda dsk, keys, **kwargs: dsk)

    def __init__(self, expr):
        self._expr = expr

    @property
    def expr(self) -> expr.Expr:
        return self._expr

    @property
    def _meta(self):
        return self.expr._meta

    @property
    def size(self):
        return new_collection(self.expr.size)

    @property
    def columns(self):
        return self._meta.columns

    def __len__(self):
        return new_collection(Len(self.expr)).compute()

    @property
    def nbytes(self):
        raise NotImplementedError("nbytes is not implemented on DataFrame")

    def __reduce__(self):
        return new_collection, (self._expr,)

    def __getitem__(self, other):
        if isinstance(other, FrameBase):
            return new_collection(self.expr.__getitem__(other.expr))
        return new_collection(self.expr.__getitem__(other))

    def __dask_graph__(self):
        out = self.expr
        out = out.optimize(fuse=False)
        return out.__dask_graph__()

    def __dask_keys__(self):
        out = self.expr
        out = out.optimize(fuse=False)
        return out.__dask_keys__()

    def simplify(self):
        return new_collection(self.expr.simplify())

    def lower_once(self):
        return new_collection(self.expr.lower_once())

    def optimize(self, combine_similar: bool = True, fuse: bool = True):
        return new_collection(
            self.expr.optimize(combine_similar=combine_similar, fuse=fuse)
        )

    @property
    def dask(self):
        return self.__dask_graph__()

    def __getattr__(self, key):
        try:
            # Prioritize `FrameBase` attributes
            return object.__getattribute__(self, key)
        except AttributeError as err:
            try:
                # Fall back to `expr` API
                # (Making sure to convert to/from Expr)
                val = getattr(self.expr, key)
                if callable(val):
                    return functools.partial(_wrap_expr_api, wrap_api=val)
                return val
            except AttributeError:
                # Raise original error
                raise err

    def compute(self, **kwargs):
        dsk = collections_to_dsk([self], True, **kwargs)
        keys = self.__dask_keys__()
        results = schedule(dsk, keys, **kwargs)
        return results

    def visualize(self, tasks: bool = False, **kwargs):
        """Visualize the expression or task graph

        Parameters
        ----------
        tasks:
            Whether to visualize the task graph. By default
            the expression graph will be visualized instead.
        """
        if tasks:
            return super().visualize(**kwargs)
        return self.expr.visualize(**kwargs)

    @property
    def index(self):
        return new_collection(self.expr.index)

    def reset_index(self, drop=False):
        return new_collection(expr.ResetIndex(self.expr, drop))

    def head(self, n=5, compute=True):
        out = new_collection(expr.Head(self.expr, n=n))
        if compute:
            out = out.compute()
        return out

    def tail(self, n=5, compute=True):
        out = new_collection(expr.Tail(self.expr, n=n))
        if compute:
            out = out.compute()
        return out

    def copy(self):
        """Return a copy of this object"""
        return new_collection(self.expr)

    def isin(self, values):
        return new_collection(expr.Isin(self.expr, values=values))

    def groupby(self, by, **kwargs):
        from pandas_expr._groupby import GroupBy

        if isinstance(by, FrameBase) and not isinstance(by, Series):
            raise ValueError(
                f"`by` must be a column name or list of columns, got {by}."
            )

        return GroupBy(self, by, **kwargs)

    def sum(self, skipna=True, numeric_only=False, min_count=0):
        return new_collection(self.expr.sum(skipna, numeric_only, min_count))

    def prod(self, skipna=True, numeric_only=False, min_count=0):
        return new_collection(self.expr.prod(skipna, numeric_only, min_count))

    def var(self, axis=0, skipna=True, ddof=1, numeric_only=False):
        return new_collection(self.expr.var(axis, skipna, ddof, numeric_only))

    def std(self, axis=0, skipna=True, ddof=1, numeric_only=False):
        return new_collection(
            self.expr.std(
                axis=axis, skipna=skipna, ddof=ddof, numeric_only=numeric_only
            )
        )

    def mean(self, skipna=True, numeric_only=False, min_count=0):
        return new_collection(self.expr.mean(skipna, numeric_only))

    def max(self, skipna=True, numeric_only=False, min_count=0):
        return new_collection(self.expr.max(skipna, numeric_only, min_count))

    def any(self, skipna=True):
        return new_collection(self.expr.any(skipna))

    def all(self, skipna=True):
        return new_collection(self.expr.all(skipna))

    def idxmin(self, skipna=True):
        return new_collection(self.expr.idxmin(skipna))

    def idxmax(self, skipna=True):
        return new_collection(self.expr.idxmax(skipna))

    def mode(self, dropna=True):
        return new_collection(self.expr.mode(dropna))

    def min(self, skipna=True, numeric_only=False, min_count=0):
        return new_collection(self.expr.min(skipna, numeric_only, min_count))

    def count(self, numeric_only=False):
        return new_collection(self.expr.count(numeric_only))

    def abs(self):
        return new_collection(self.expr.abs())

    def astype(self, dtypes):
        return new_collection(self.expr.astype(dtypes))

    def clip(self, lower=None, upper=None):
        return new_collection(self.expr.clip(lower, upper))

    def combine_first(self, other):
        return new_collection(self.expr.combine_first(other.expr))

    def to_timestamp(self, freq=None, how="start"):
        return new_collection(self.expr.to_timestamp(freq, how))

    def isna(self):
        return new_collection(self.expr.isna())

    def round(self, decimals=0):
        return new_collection(self.expr.round(decimals))

    def apply(self, function, *args, **kwargs):
        return new_collection(self.expr.apply(function, *args, **kwargs))

    def replace(self, to_replace=None, value=no_default, regex=False):
        return new_collection(self.expr.replace(to_replace, value, regex))

    def fillna(self, value=None):
        return new_collection(self.expr.fillna(value))

    def rename_axis(
        self, mapper=no_default, index=no_default, columns=no_default, axis=0
    ):
        return new_collection(self.expr.rename_axis(mapper, index, columns, axis))

    def align(self, other, join="outer", fill_value=None):
        return self.expr.align(other.expr, join, fill_value)

    def nunique_approx(self):
        return new_collection(self.expr.nunique_approx())


# Add operator attributes
for op in [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__lt__",
    "__rlt__",
    "__gt__",
    "__rgt__",
    "__le__",
    "__rle__",
    "__ge__",
    "__rge__",
    "__eq__",
    "__ne__",
    "__and__",
    "__rand__",
    "__or__",
    "__ror__",
    "__xor__",
    "__rxor__",
]:
    setattr(FrameBase, op, functools.partialmethod(_wrap_expr_op, op=op))

for op in [
    "__invert__",
    "__neg__",
    "__pos__",
]:
    setattr(FrameBase, op, functools.partialmethod(_wrap_unary_expr_op, op=op))


class DataFrame(FrameBase):
    """DataFrame-like Expr Collection"""

    def __init__(
        self,
        data=None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
    ) -> None:
        if isinstance(data, expr.Expr):
            super().__init__(data)
        else:
            from pandas_expr.io import PandasIO

            super().__init__(PandasIO(data, index, columns, dtype, copy))

    def assign(self, **pairs):
        result = self
        data = self.copy()
        for key, value in pairs.items():
            if callable(value):
                value = value(data)

            if isinstance(value, FrameBase):
                value = value.expr

            result = new_collection(expr.Assign(result.expr, key, value))
        return result

    def merge(
        self,
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        suffixes=("_x", "_y"),
        indicator=False,
    ):
        if on is not None:
            left_on = right_on = on
        return new_collection(
            Merge(
                self.expr,
                right.expr,
                how=how,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
                suffixes=suffixes,
                indicator=indicator,
            )
        )

    def join(
        self,
        other,
        on=None,
        how="left",
        lsuffix="",
        rsuffix="",
        shuffle_backend=None,
    ):
        if (
            not isinstance(other, list)
            and not isinstance(other._meta, pd.DataFrame)
            and hasattr(other._meta, "name")
        ):
            other = new_collection(expr.ToFrame(other.expr))

        if not isinstance(other, FrameBase):
            return new_collection(
                JoinRecursive([self.expr] + [o.expr for o in other], how=how)
            )

        return self.merge(
            right=other,
            left_index=on is None,
            right_index=True,
            left_on=on,
            how=how,
            suffixes=(lsuffix, rsuffix),
        )

    def __setitem__(self, key, value):
        out = self.assign(**{key: value})
        self._expr = out._expr

    def __delitem__(self, key):
        columns = [c for c in self.columns if c != key]
        out = self[columns]
        self._expr = out._expr

    def __getattr__(self, key):
        try:
            # Prioritize `DataFrame` attributes
            return object.__getattribute__(self, key)
        except AttributeError as err:
            try:
                # Check if key is in columns if key
                # is not a normal attribute
                if key in self.expr._meta.columns:
                    return Series(self.expr[key])
                raise err
            except AttributeError:
                # Fall back to `BaseFrame.__getattr__`
                return super().__getattr__(key)

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(set(dir(expr.Expr)))
        o.update(c for c in self.columns if (isinstance(c, str) and c.isidentifier()))
        return list(o)

    def map(self, func, na_action=None):
        return new_collection(expr.Map(self.expr, arg=func, na_action=na_action))

    def __repr__(self):
        return f"<pandas_expr.expr.DataFrame: expr={self.expr}>"

    def nlargest(self, n=5, columns=None):
        return new_collection(NLargest(self.expr, n=n, _columns=columns))

    def nsmallest(self, n=5, columns=None):
        return new_collection(NSmallest(self.expr, n=n, _columns=columns))

    def memory_usage(self, deep=False, index=True):
        return new_collection(MemoryUsageFrame(self.expr, deep=deep, _index=index))

    def drop_duplicates(self, subset=None, ignore_index=False):
        subset = _convert_to_list(subset)
        return new_collection(
            DropDuplicates(self.expr, subset=subset, ignore_index=ignore_index)
        )

    def dropna(self, how=no_default, subset=None, thresh=no_default):
        subset = _convert_to_list(subset)
        return new_collection(
            expr.DropnaFrame(self.expr, how=how, subset=subset, thresh=thresh)
        )

    def rename(self, columns):
        return new_collection(expr.RenameFrame(self.expr, columns=columns))

    def explode(self, column):
        column = _convert_to_list(column)
        return new_collection(expr.ExplodeFrame(self.expr, column=column))

    def drop(self, labels=None, columns=None, errors="raise"):
        if columns is None:
            columns = labels
        if columns is None:
            raise TypeError("must either specify 'columns' or 'labels'")
        return new_collection(expr.Drop(self.expr, columns=columns, errors=errors))

    def to_parquet(self, path, **kwargs):
        from pandas_expr.io.parquet import to_parquet

        return to_parquet(self, path, **kwargs)

    def select_dtypes(self, include=None, exclude=None):
        columns = self._meta.select_dtypes(include=include, exclude=exclude).columns
        return new_collection(self.expr[columns])

    def eval(self, expr, **kwargs):
        return new_collection(Eval(self.expr, _expr=expr, expr_kwargs=kwargs))

    def set_index(self, other, drop=True):
        return new_collection(expr.SetIndex(self.expr, other, drop))


class Series(FrameBase):
    """Series-like Expr Collection"""

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(set(dir(expr.Expr)))
        return list(o)

    @property
    def name(self):
        return self.expr._meta.name

    @property
    def nbytes(self):
        return new_collection(self.expr.nbytes)

    def map(self, arg, na_action=None):
        return new_collection(expr.Map(self.expr, arg=arg, na_action=na_action))

    def __repr__(self):
        return f"<pandas_expr.expr.Series: expr={self.expr}>"

    def to_frame(self, name=no_default):
        return new_collection(expr.ToFrame(self.expr, name=name))

    def value_counts(self, sort=None, ascending=False, dropna=True, normalize=False):
        return new_collection(
            ValueCounts(self.expr, sort, ascending, dropna, normalize)
        )

    def nlargest(self, n=5):
        return new_collection(NLargest(self.expr, n=n))

    def nsmallest(self, n=5):
        return new_collection(NSmallest(self.expr, n=n))

    def memory_usage(self, deep=False, index=True):
        return new_collection(MemoryUsageFrame(self.expr, deep=deep, _index=index))

    def unique(self):
        return new_collection(Unique(self.expr))

    def drop_duplicates(self, ignore_index=False):
        return new_collection(DropDuplicates(self.expr, ignore_index=ignore_index))

    def dropna(self):
        return new_collection(expr.DropnaSeries(self.expr))

    def between(self, left, right, inclusive="both"):
        return new_collection(
            expr.Between(self.expr, left=left, right=right, inclusive=inclusive)
        )

    def explode(self):
        return new_collection(expr.ExplodeSeries(self.expr))

    _accessors = {"cat"}
    cat = CachedAccessor("cat", CategoricalAccessor)


class Index(Series):
    """Index-like Expr Collection"""

    def __repr__(self):
        return f"<pandas_expr.expr.Index: expr={self.expr}>"

    def to_frame(self, index=True, name=no_default):
        if not index:
            raise NotImplementedError
        return new_collection(expr.ToFrameIndex(self.expr, index=index, name=name))

    def memory_usage(self, deep=False):
        return new_collection(MemoryUsageIndex(self.expr, deep=deep))

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(set(dir(expr.Expr)))
        return list(o)


class Scalar(FrameBase):
    """Scalar Expr Collection"""

    def __repr__(self):
        return f"<pandas_expr.expr.Scalar: expr={self.expr}>"


def new_collection(expr):
    """Create new collection from an expr"""

    meta = expr._meta
    if isinstance(meta, pd.DataFrame):
        return DataFrame(expr)
    elif isinstance(meta, pd.Series):
        return Series(expr)
    elif isinstance(meta, pd.Index):
        return Index(expr)
    else:
        return Scalar(expr)


def optimize(collection, fuse=True):
    return new_collection(expr.optimize(collection.expr, fuse=fuse))


def read_csv(path, *args, **kwargs):
    from pandas_expr.io.csv import ReadCSV

    if not isinstance(path, str):
        path = stringify_path(path)
    return new_collection(ReadCSV(path, *args, **kwargs))


def read_parquet(
    path=None,
    columns=None,
    filters=None,
    categories=None,
    index=None,
    storage_options=None,
    dtype_backend=None,
    calculate_divisions=False,
    ignore_metadata_file=False,
    metadata_task_size=None,
    split_row_groups="infer",
    blocksize="default",
    aggregate_files=None,
    parquet_file_extension=(".parq", ".parquet", ".pq"),
    filesystem="fsspec",
    **kwargs,
):
    from pandas_expr.io.parquet import ReadParquet

    if not isinstance(path, str):
        path = stringify_path(path)

    kwargs["dtype_backend"] = dtype_backend

    return new_collection(
        ReadParquet(
            path,
            columns=_convert_to_list(columns),
            filters=filters,
            categories=categories,
            index=index,
            storage_options=storage_options,
            calculate_divisions=calculate_divisions,
            ignore_metadata_file=ignore_metadata_file,
            metadata_task_size=metadata_task_size,
            split_row_groups=split_row_groups,
            blocksize=blocksize,
            aggregate_files=aggregate_files,
            parquet_file_extension=parquet_file_extension,
            filesystem=filesystem,
            kwargs=kwargs,
        )
    )


def concat(
    dfs,
    axis=0,
    join="outer",
):
    if axis == 1:
        # TODO: implement
        raise NotImplementedError

    return new_collection(
        Concat(
            join,
            *[df.expr if isinstance(df, FrameBase) else df for df in dfs],
        )
    )
