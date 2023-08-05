from __future__ import annotations

import itertools

import pandas as pd
import pyarrow.parquet as pp

from pandas_expr._expr import (
    EQ,
    GE,
    GT,
    LE,
    LT,
    NE,
    And,
    Expr,
    Filter,
    Index,
    Lengths,
    Literal,
    Or,
    Projection,
)
from pandas_expr._util import M, _convert_to_list
from pandas_expr.io import BlockwiseIO

NONE_LABEL = "__null_dask_index__"

# TODO: Allow _cached_dataset_info/_plan to contain >1 item?
_cached_dataset_info = {}
_cached_plan = {}

#
# @normalize_token.register(pa_ds.Dataset)
# def normalize_pa_ds(ds):
#     return (ds.files, ds.schema)
#
#
# @normalize_token.register(pa_ds.FileFormat)
# def normalize_pa_file_format(file_format):
#     return str(file_format)
#
#
# @normalize_token.register(pa.Schema)
# def normalize_pa_schema(schema):
#     return schema.to_string()


class ToParquet(Expr):
    _parameters = [
        "frame",
        "path",
    ]

    @property
    def _meta(self):
        return None

    def _task(self):
        return (M.to_parquet,) + tuple(self.operands)


def to_parquet(
    df,
    *args,
    **kwargs,
):
    from pandas_expr._collection import new_collection

    return new_collection(
        ToParquet(
            df.expr,
            *args,
            **kwargs,
        )
    )


class ReadParquet(BlockwiseIO):
    """Read a parquet dataset"""

    _parameters = [
        "path",
        "columns",
        "filters",
        "storage_options",
        "dtype_backend",
        "_series",
    ]
    _defaults = {
        "columns": None,
        "filters": None,
        "storage_options": None,
        "dtype_backend": pd.api.extensions.no_default,
        "_series": False,
    }
    _keyword_only = [
        "columns",
        "filters",
        "storage_options",
        "dtype_backend",
        "_series",
    ]

    def operation(self, *args, _series=None, **kwargs):
        result = pd.read_parquet(*args, **kwargs)
        if _series:
            return result[result.columns[0]]
        return result

    @property
    def engine(self):
        return "pyarrow"

    @property
    def columns(self):
        columns_operand = self.operand("columns")
        if columns_operand is None:
            return list(self._meta.columns)
        else:
            return _convert_to_list(columns_operand)

    def _combine_similar(self, root: Expr):
        # For ReadParquet, we can avoid redundant file-system
        # access by aggregating multiple operations with different
        # column projections into the same operation.
        alike = self._find_similar_operations(root, ignore=["columns", "_series"])
        if alike:
            # We have other ReadParquet operations in the expression
            # graph that can be combined with this one.

            # Find the column-projection union needed to combine
            # the qualified ReadParquet operations
            columns = set()
            rps = [self] + alike
            for rp in rps:
                if rp.operand("columns"):
                    columns |= set(rp.operand("columns"))
            columns = sorted(columns)

            # Can bail if we are not changing columns or the "_series" operand
            columns_operand = self.operand("columns")
            if columns_operand == columns and (len(columns) > 1 or not self._series):
                return

            # Check if we have the operation we want elsewhere in the graph
            for rp in rps:
                if rp.operand("columns") == columns and not rp.operand("_series"):
                    return (
                        rp[columns_operand[0]] if self._series else rp[columns_operand]
                    )

            # Create the "combined" ReadParquet operation
            subs = {"columns": columns}
            if self._series:
                subs["_series"] = False
            new = self.substitute_parameters(subs)
            return new[columns_operand[0]] if self._series else new[columns_operand]

        return

    def _simplify_up(self, parent):
        if isinstance(parent, Index):
            # Column projection
            return self.substitute_parameters({"columns": [], "_series": False})

        if isinstance(parent, Projection):
            # Column projection
            parent_columns = parent.operand("columns")
            substitutions = {"columns": _convert_to_list(parent_columns)}
            if isinstance(parent_columns, (str, int)):
                substitutions["_series"] = True
            return self.substitute_parameters(substitutions)

        if isinstance(parent, Filter) and isinstance(
            parent.predicate, (LE, GE, LT, GT, EQ, NE, And, Or)
        ):
            # Predicate pushdown
            filters = _DNF.extract_pq_filters(self, parent.predicate)
            if filters:
                kwargs = dict(zip(self._parameters, self.operands))
                kwargs["filters"] = filters.combine(kwargs["filters"]).to_list_tuple()
                return ReadParquet(**kwargs)

        if isinstance(parent, Lengths):
            _lengths = self._get_lengths()
            if _lengths:
                return Literal(_lengths)

    @property
    def _meta(self):
        df = pp.read_schema(self.path).empty_table().to_pandas()
        if self.operand("columns") is not None:
            df = df[self.operand("columns")]
        if self._series:
            return df[df.columns[0]]
        return df


#
# Filtering logic
#


class _DNF:
    """Manage filters in Disjunctive Normal Form (DNF)"""

    class _Or(frozenset):
        """Fozen set of disjunctions"""

        def to_list_tuple(self) -> list:
            # DNF "or" is List[List[Tuple]]
            def _maybe_list(val):
                if isinstance(val, tuple) and val and isinstance(val[0], (tuple, list)):
                    return list(val)
                return [val]

            return [
                _maybe_list(val.to_list_tuple())
                if hasattr(val, "to_list_tuple")
                else _maybe_list(val)
                for val in self
            ]

    class _And(frozenset):
        """Frozen set of conjunctions"""

        def to_list_tuple(self) -> list:
            # DNF "and" is List[Tuple]
            return tuple(
                val.to_list_tuple() if hasattr(val, "to_list_tuple") else val
                for val in self
            )

    _filters: _And | _Or | None  # Underlying filter expression

    def __init__(self, filters: _And | _Or | list | tuple | None) -> _DNF:
        self._filters = self.normalize(filters)

    def to_list_tuple(self) -> list:
        return self._filters.to_list_tuple()

    def __bool__(self) -> bool:
        return bool(self._filters)

    @classmethod
    def normalize(cls, filters: _And | _Or | list | tuple | None):
        """Convert raw filters to the `_Or(_And)` DNF representation"""
        if not filters:
            result = None
        elif isinstance(filters, list):
            conjunctions = filters if isinstance(filters[0], list) else [filters]
            result = cls._Or([cls._And(conjunction) for conjunction in conjunctions])
        elif isinstance(filters, tuple):
            if isinstance(filters[0], tuple):
                raise TypeError("filters must be List[Tuple] or List[List[Tuple]]")
            result = cls._Or((cls._And((filters,)),))
        elif isinstance(filters, cls._Or):
            result = cls._Or(se for e in filters for se in cls.normalize(e))
        elif isinstance(filters, cls._And):
            total = []
            for c in itertools.product(*[cls.normalize(e) for e in filters]):
                total.append(cls._And(se for e in c for se in e))
            result = cls._Or(total)
        else:
            raise TypeError(f"{type(filters)} not a supported type for _DNF")
        return result

    def combine(self, other: _DNF | _And | _Or | list | tuple | None) -> _DNF:
        """Combine with another _DNF object"""
        if not isinstance(other, _DNF):
            other = _DNF(other)
        assert isinstance(other, _DNF)
        if self._filters is None:
            result = other._filters
        elif other._filters is None:
            result = self._filters
        else:
            result = self._And([self._filters, other._filters])
        return _DNF(result)

    @classmethod
    def extract_pq_filters(cls, pq_expr: ReadParquet, predicate_expr: Expr) -> _DNF:
        _filters = None
        if isinstance(predicate_expr, (LE, GE, LT, GT, EQ, NE)):
            if (
                isinstance(predicate_expr.left, ReadParquet)
                and predicate_expr.left.path == pq_expr.path
                and not isinstance(predicate_expr.right, Expr)
            ):
                op = predicate_expr._operator_repr
                column = predicate_expr.left.columns[0]
                value = predicate_expr.right
                _filters = (column, op, value)
            elif (
                isinstance(predicate_expr.right, ReadParquet)
                and predicate_expr.right.path == pq_expr.path
                and not isinstance(predicate_expr.left, Expr)
            ):
                # Simple dict to make sure field comes first in filter
                flip = {LE: GE, LT: GT, GE: LE, GT: LT}
                op = predicate_expr
                op = flip.get(op, op)._operator_repr
                column = predicate_expr.right.columns[0]
                value = predicate_expr.left
                _filters = (column, op, value)

        elif isinstance(predicate_expr, (And, Or)):
            left = cls.extract_pq_filters(pq_expr, predicate_expr.left)._filters
            right = cls.extract_pq_filters(pq_expr, predicate_expr.right)._filters
            if left and right:
                if isinstance(predicate_expr, And):
                    _filters = cls._And([left, right])
                else:
                    _filters = cls._Or([left, right])

        return _DNF(_filters)
