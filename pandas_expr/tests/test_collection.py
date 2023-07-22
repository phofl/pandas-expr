import operator
import re

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from dask.dataframe.utils import UNKNOWN_CATEGORIES, assert_eq

from pandas_expr import DataFrame, expr, optimize

# from pandas_expr._reductions import Len
from pandas_expr._util import M


@pytest.fixture
def pdf():
    pdf = pd.DataFrame({"x": range(100)})
    pdf["y"] = pdf.x * 10.0
    yield pdf


@pytest.fixture
def df(pdf):
    yield DataFrame(pdf)


def test_del(pdf, df):
    pdf = pdf.copy()

    # Check __delitem__
    del pdf["x"]
    del df["x"]
    assert_eq(pdf, df)


def test_setitem(pdf, df):
    pdf = pdf.copy()
    pdf["z"] = pdf.x + pdf.y

    df["z"] = df.x + df.y

    assert "z" in df.columns
    assert_eq(df, pdf)


def test_explode():
    pdf = pd.DataFrame({"a": [[1, 2], [3, 4]]})
    df = DataFrame(pdf)
    assert_eq(pdf.explode(column="a"), df.explode(column="a"))
    assert_eq(pdf.a.explode(), df.a.explode())


def test_explode_simplify(pdf):
    pdf["z"] = 1
    df = DataFrame(pdf)
    q = df.explode(column="x")["y"]
    result = optimize(q, fuse=False)
    expected = df[["x", "y"]].explode(column="x")["y"]
    assert result._name == expected._name


def test_meta_blockwise():
    a = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    b = pd.DataFrame({"z": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})

    aa = DataFrame(a)
    bb = DataFrame(b)

    cc = 2 * aa - 3 * bb
    assert set(cc.columns) == {"x", "y", "z"}


def test_dask(pdf, df):
    z = (df.x + df.y).sum()

    assert assert_eq(z, (pdf.x + pdf.y).sum())


@pytest.mark.parametrize(
    "func",
    [
        M.max,
        M.min,
        M.any,
        M.all,
        M.sum,
        M.prod,
        M.count,
        M.mean,
        M.std,
        M.var,
        M.idxmin,
        M.idxmax,
        pytest.param(
            lambda df: df.size,
            marks=pytest.mark.skip(reason="scalars don't work yet"),
        ),
    ],
)
def test_reductions(func, pdf, df):
    result = func(df)
    assert_eq(result, func(pdf))
    result = func(df.x)
    assert_eq(result, func(pdf.x))
    # check_dtype False because sub-selection of columns that is pushed through
    # is not reflected in the meta calculation
    assert_eq(func(df)["x"], func(pdf)["x"], check_dtype=False)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("ddof", [1, 2])
def test_std_kwargs(axis, skipna, ddof):
    pdf = pd.DataFrame(
        {"x": range(30), "y": [1, 2, None] * 10, "z": ["dog", "cat"] * 15}
    )
    df = DataFrame(pdf)
    assert_eq(
        pdf.std(axis=axis, skipna=skipna, ddof=ddof, numeric_only=True),
        df.std(axis=axis, skipna=skipna, ddof=ddof, numeric_only=True),
    )


def test_nbytes(pdf, df):
    with pytest.raises(NotImplementedError, match="nbytes is not implemented"):
        df.nbytes
    assert_eq(df.x.nbytes, pdf.x.nbytes)


def test_mode():
    pdf = pd.DataFrame({"x": [1, 2, 3, 1, 2]})
    df = DataFrame(pdf)

    assert_eq(df.x.mode(), pdf.x.mode(), check_names=False)


def test_value_counts(df, pdf):
    with pytest.raises(
        AttributeError, match="'DataFrame' object has no attribute 'value_counts'"
    ):
        df.value_counts()
    assert_eq(df.x.value_counts(), pdf.x.value_counts())


def test_dropna(pdf):
    pdf.loc[0, "y"] = np.nan
    df = DataFrame(pdf)
    assert_eq(df.dropna(), pdf.dropna())
    assert_eq(df.dropna(how="all"), pdf.dropna(how="all"))
    assert_eq(df.y.dropna(), pdf.y.dropna())


def test_fillna():
    pdf = pd.DataFrame({"x": [1, 2, None, None, 5, 6]})
    df = DataFrame(pdf)
    actual = df.fillna(value=100)
    expected = pdf.fillna(value=100)
    assert_eq(actual, expected)


def test_memory_usage(pdf):
    # Results are not equal with RangeIndex because pandas has one RangeIndex while
    # we have one RangeIndex per partition
    pdf.index = np.arange(len(pdf))
    df = DataFrame(pdf)
    assert_eq(df.memory_usage(), pdf.memory_usage())
    assert_eq(df.memory_usage(index=False), pdf.memory_usage(index=False))
    assert_eq(df.x.memory_usage(), pdf.x.memory_usage())
    assert_eq(df.x.memory_usage(index=False), pdf.x.memory_usage(index=False))
    assert_eq(df.index.memory_usage(), pdf.index.memory_usage())
    with pytest.raises(TypeError, match="got an unexpected keyword"):
        df.index.memory_usage(index=True)


@pytest.mark.parametrize("func", [M.nlargest, M.nsmallest])
def test_nlargest_nsmallest(df, pdf, func):
    assert_eq(func(df, n=5, columns="x"), func(pdf, n=5, columns="x"))
    assert_eq(func(df.x, n=5), func(pdf.x, n=5))
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        func(df.x, n=5, columns="foo")


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.x > 10,
        lambda df: df.x + 20 > df.y,
        lambda df: 10 < df.x,
        lambda df: 10 <= df.x,
        lambda df: 10 == df.x,
        lambda df: df.x < df.y,
        lambda df: df.x > df.y,
        lambda df: df.x == df.y,
        lambda df: df.x != df.y,
    ],
)
def test_conditionals(func, pdf, df):
    assert_eq(func(pdf), func(df), check_names=False)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.x & df.y,
        lambda df: df.x.__rand__(df.y),
        lambda df: df.x | df.y,
        lambda df: df.x.__ror__(df.y),
        lambda df: df.x ^ df.y,
        lambda df: df.x.__rxor__(df.y),
    ],
)
def test_boolean_operators(func):
    pdf = pd.DataFrame(
        {"x": [True, False, True, False], "y": [True, False, False, False]}
    )
    df = DataFrame(pdf)
    assert_eq(func(pdf), func(df))


@pytest.mark.parametrize(
    "func",
    [
        lambda df: ~df,
        lambda df: ~df.x,
        lambda df: -df.z,
        lambda df: +df.z,
        lambda df: -df,
        lambda df: +df,
    ],
)
def test_unary_operators(func):
    pdf = pd.DataFrame(
        {"x": [True, False, True, False], "y": [True, False, False, False], "z": 1}
    )
    df = DataFrame(pdf)
    assert_eq(func(pdf), func(df))


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df[(df.x > 10) | (df.x < 5)],
        lambda df: df[(df.x > 7) & (df.x < 10)],
    ],
)
def test_and_or(func, pdf, df):
    assert_eq(func(pdf), func(df), check_names=False)


@pytest.mark.parametrize("how", ["start", "end"])
def test_to_timestamp(pdf, how):
    pdf.index = pd.period_range("2019-12-31", freq="D", periods=len(pdf))
    df = DataFrame(pdf)
    assert_eq(df.to_timestamp(how=how), pdf.to_timestamp(how=how))
    assert_eq(df.x.to_timestamp(how=how), pdf.x.to_timestamp(how=how))


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.astype(int),
        lambda df: df.apply(lambda row, x, y=10: row * x + y, x=2),
        lambda df: df.clip(lower=10, upper=50),
        lambda df: df.x.clip(lower=10, upper=50),
        lambda df: df.x.between(left=10, right=50),
        lambda df: df.x.map(lambda x: x + 1),
        lambda df: df.index.map(lambda x: x + 1),
        lambda df: df[df.x > 5],
        lambda df: df.assign(a=df.x + df.y, b=df.x - df.y),
        lambda df: df.replace(to_replace=1, value=1000),
        lambda df: df.x.replace(to_replace=1, value=1000),
        lambda df: df.isna(),
        lambda df: df.x.isna(),
        lambda df: df.abs(),
        lambda df: df.x.abs(),
        lambda df: df.rename(columns={"x": "xx"}),
        lambda df: df.rename(columns={"x": "xx"}).xx,
        lambda df: df.rename(columns={"x": "xx"})[["xx"]],
        lambda df: df.combine_first(df),
        lambda df: df.x.combine_first(df.y),
        lambda df: df.x.to_frame(),
        lambda df: df.drop(columns="x"),
        lambda df: df.x.index.to_frame(),
        lambda df: df.eval("z=x+y"),
        lambda df: df.select_dtypes(include="integer"),
    ],
)
def test_blockwise(func, pdf, df):
    assert_eq(func(pdf), func(df))


def test_rename_axis(pdf):
    pdf.index.name = "a"
    pdf.columns.name = "b"
    df = DataFrame(pdf)
    assert_eq(df.rename_axis(index="dummy"), pdf.rename_axis(index="dummy"))
    assert_eq(df.rename_axis(columns="dummy"), pdf.rename_axis(columns="dummy"))
    assert_eq(df.x.rename_axis(index="dummy"), pdf.x.rename_axis(index="dummy"))


def test_isin(df, pdf):
    values = [1, 2]
    assert_eq(pdf.isin(values), df.isin(values))
    assert_eq(pdf.x.isin(values), df.x.isin(values))


def test_round(pdf):
    pdf += 0.5555
    df = DataFrame(pdf)
    assert_eq(df.round(decimals=1), pdf.round(decimals=1))
    assert_eq(df.x.round(decimals=1), pdf.x.round(decimals=1))


def test_repr(df):
    assert "+ 1" in str(df + 1)
    assert "+ 1" in repr(df + 1)

    s = (df["x"] + 1).sum(skipna=False).expr
    assert '["x"]' in s or "['x']" in s
    assert "+ 1" in s
    assert "sum(skipna=False)" in s


def test_combine_first_simplify(pdf):
    df = DataFrame(pdf)
    pdf2 = pdf.rename(columns={"y": "z"})
    df2 = DataFrame(pdf2)

    q = df.combine_first(df2)[["z", "y"]]
    result = q.simplify()
    expected = df[["y"]].combine_first(df2[["z"]])[["z", "y"]]
    assert result._name == expected._name
    assert_eq(result, pdf.combine_first(pdf2)[["z", "y"]])


def test_rename_traverse_filter(df):
    result = df.rename(columns={"x": "xx"})[["xx"]].simplify()
    expected = df[["x"]].rename(columns={"x": "xx"})
    assert str(result) == str(expected)


def test_columns_traverse_filters(pdf, df):
    result = df[df.x > 5].y.simplify()
    expected = df.y[df.x > 5]

    assert str(result) == str(expected)


def test_clip_traverse_filters(df):
    result = df.clip(lower=10).y.simplify()
    expected = df.y.clip(lower=10)

    assert result._name == expected._name

    result = df.clip(lower=10)[["x", "y"]].simplify()
    expected = df.clip(lower=10)

    assert result._name == expected._name

    arg = df.clip(lower=10)[["x"]]
    result = arg.simplify()
    expected = df[["x"]].clip(lower=10)

    assert result._name == expected._name


@pytest.mark.parametrize("projection", ["zz", ["zz"], ["zz", "x"], "zz"])
@pytest.mark.parametrize("subset", ["x", ["x"]])
def test_drop_duplicates_subset_simplify(pdf, subset, projection):
    pdf["zz"] = 1
    df = DataFrame(pdf)
    result = df.drop_duplicates(subset=subset)[projection].simplify()
    expected = df[["x", "zz"]].drop_duplicates(subset=subset)[projection]

    assert str(result) == str(expected)


def test_broadcast(pdf, df):
    assert_eq(
        df + df.sum(),
        pdf + pdf.sum(),
    )
    assert_eq(
        df.x + df.x.sum(),
        pdf.x + pdf.x.sum(),
    )


def test_index(pdf, df):
    assert_eq(df.index, pdf.index)
    assert_eq(df.x.index, pdf.x.index)


@pytest.mark.parametrize("drop", [True, False])
def test_reset_index(pdf, df, drop):
    assert_eq(df.reset_index(drop=drop), pdf.reset_index(drop=drop), check_index=False)
    assert_eq(
        df.x.reset_index(drop=drop), pdf.x.reset_index(drop=drop), check_index=False
    )


def test_head(pdf, df):
    assert_eq(df.head(compute=False), pdf.head())
    assert_eq(df.head(compute=False, n=7), pdf.head(n=7))


def test_head_down(df):
    result = (df.x + df.y + 1).head(compute=False)
    optimized = result.simplify()

    assert_eq(result, optimized)

    assert not isinstance(optimized.expr, expr.Head)


def test_head_head(df):
    a = df.head(compute=False).head(compute=False)
    b = df.head(compute=False)

    assert a.optimize()._name == b.optimize()._name


def test_tail(pdf, df):
    assert_eq(df.tail(compute=False), pdf.tail())
    assert_eq(df.tail(compute=False, n=7), pdf.tail(n=7))


def test_tail_down(df):
    result = (df.x + df.y + 1).tail(compute=False)
    optimized = optimize(result, fuse=False)

    assert_eq(result, optimized)

    assert not isinstance(optimized.expr, expr.Tail)


def test_tail_tail(df):
    a = df.tail(compute=False).tail(compute=False)
    b = df.tail(compute=False)

    assert a.optimize()._name == b.optimize()._name


def test_projection_stacking(df):
    result = df[["x", "y"]]["x"]
    optimized = result.simplify()
    expected = df["x"]

    assert optimized._name == expected._name


def test_projection_stacking_coercion(pdf):
    df = DataFrame(pdf)
    assert_eq(df.x[0], pdf.x[0], check_divisions=False)
    assert_eq(df.x[[0]], pdf.x[[0]], check_divisions=False)


def test_remove_unnecessary_projections(df):
    result = (df + 1)[df.columns]
    optimized = result.simplify()
    expected = df + 1

    assert optimized._name == expected._name

    result = (df[["x"]] + 1)[["x"]]
    optimized = result.simplify()
    expected = df[["x"]] + 1

    assert optimized._name == expected._name


def test_substitute(df):
    pdf = pd.DataFrame(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
        }
    )
    df = DataFrame(pdf)
    df = df.expr

    result = (df + 1).substitute({1: 2})
    expected = df + 2
    assert result._name == expected._name

    result = df["a"].substitute({df["a"]: df["b"]})
    expected = df["b"]
    assert result._name == expected._name

    result = (df["a"] - df["b"]).substitute({df["b"]: df["c"]})
    expected = df["a"] - df["c"]
    assert result._name == expected._name

    result = df["a"].substitute({3: 4})
    expected = DataFrame(pdf).a
    assert result._name == expected._name

    result = (df["a"].sum() + 5).substitute({df["a"]: df["b"], 5: 6})
    expected = df["b"].sum() + 6
    assert result._name == expected._name


def test_substitute_parameters(df):
    pdf = pd.DataFrame(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
            "index": range(100, 0, -1),
        }
    )
    df = DataFrame(pdf)

    df2 = df[["a", "b"]]
    result = df2.substitute_parameters({"columns": "c"})
    assert result._name != df2._name
    assert result._name == df["c"]._name


def test_copy(pdf, df):
    original = df.copy()
    columns = tuple(original.columns)

    df["z"] = df.x + df.y

    assert tuple(original.columns) == columns
    assert "z" not in original.columns


def test_column_getattr(df):
    df = df.expr
    assert df.x._name == df["x"]._name

    with pytest.raises(AttributeError):
        df.foo


# def test_size_optimized(df):
#     expr = (df.x + 1).apply(lambda x: x).size
#     out = optimize(expr)
#     expected = optimize(df.x.size)
#     assert out._name == expected._name
#
#     expr = (df + 1).apply(lambda x: x).size
#     out = optimize(expr)
#     expected = optimize(df.size)
#     assert out._name == expected._name


def test_simple_graphs(df):
    expr = (df + 1).expr
    graph = expr.__dask_graph__()

    assert graph[expr._name] == (operator.add, (df.expr._name, 0), 1)


def test_depth(df):
    assert df._depth() == 1
    assert (df + 1)._depth() == 2
    assert ((df.x + 1) + df.y)._depth() == 4


# def test_len(df, pdf):
#     df2 = df[["x"]] + 1
#     assert len(df2) == len(pdf)
#
#     assert len(df[df.x > 5]) == len(pdf[pdf.x > 5])
#
#     first = df2.partitions[0].compute()
#     assert len(df2.partitions[0]) == len(first)
#
#     assert isinstance(Len(df2.expr).optimize(), expr.Literal)
#     assert isinstance(expr.Lengths(df2.expr).optimize(), expr.Literal)


def test_astype_simplify(df, pdf):
    q = df.astype({"x": "float64", "y": "float64"})["x"]
    result = q.simplify()
    expected = df["x"].astype({"x": "float64"})
    assert result._name == expected._name
    assert_eq(q, pdf.astype({"x": "float64", "y": "float64"})["x"])

    q = df.astype({"y": "float64"})["x"]
    result = q.simplify()
    expected = df["x"]
    assert result._name == expected._name

    q = df.astype("float64")["x"]
    result = q.simplify()
    expected = df["x"].astype("float64")
    assert result._name == expected._name


def test_drop_duplicates(df, pdf):
    assert_eq(df.drop_duplicates(), pdf.drop_duplicates())
    assert_eq(
        df.drop_duplicates(ignore_index=True), pdf.drop_duplicates(ignore_index=True)
    )
    assert_eq(df.drop_duplicates(subset=["x"]), pdf.drop_duplicates(subset=["x"]))
    assert_eq(df.x.drop_duplicates(), pdf.x.drop_duplicates())

    with pytest.raises(KeyError, match=re.escape("Index(['a'], dtype='object')")):
        df.drop_duplicates(subset=["a"])

    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        df.x.drop_duplicates(subset=["a"])


@pytest.mark.xfail(reason="figure out how to handle methods that return arrays")
def test_unique(df, pdf):
    with pytest.raises(
        AttributeError, match="'DataFrame' object has no attribute 'unique'"
    ):
        df.unique()

    tm.assert_numpy_array_equal(df.x.unique(), pdf.x.unique())
    assert_eq(df.index.unique(), pd.Index(pdf.index.unique()))


def test_walk(df):
    df2 = df[df["x"] > 1][["y"]] + 1
    assert all(isinstance(ex, expr.Expr) for ex in df2.walk())
    exprs = set(df2.walk())
    assert df.expr in exprs
    assert df["x"].expr in exprs
    assert (df["x"] > 1).expr in exprs
    assert 1 not in exprs


def test_find_operations(df):
    df2 = df[df["x"] > 1][["y"]] + 1

    filters = list(df2.find_operations(expr.Filter))
    assert len(filters) == 1

    projections = list(df2.find_operations(expr.Projection))
    assert len(projections) == 2

    adds = list(df2.find_operations(expr.Add))
    assert len(adds) == 1
    assert next(iter(adds))._name == df2._name

    both = list(df2.find_operations((expr.Add, expr.Filter)))
    assert len(both) == 2


@pytest.mark.parametrize("subset", ["x", ["x"]])
def test_dropna_simplify(pdf, subset):
    pdf["z"] = 1
    df = DataFrame(pdf)
    q = df.dropna(subset=subset)["y"]
    result = q.simplify()
    expected = df[["x", "y"]].dropna(subset=subset)["y"]
    assert result._name == expected._name
    assert_eq(q, pdf.dropna(subset=subset)["y"])


def test_dir(df):
    assert all(c in dir(df) for c in df.columns)
    assert "sum" in dir(df)
    assert "sum" in dir(df.x)
    assert "sum" in dir(df.index)


@pytest.mark.parametrize(
    "func, args",
    [
        ("replace", (1, 2)),
        ("isin", ([1, 2],)),
        ("clip", (0, 5)),
        ("isna", ()),
        ("round", ()),
        ("abs", ()),
        ("fillna", ({"x": 1})),
        # ("map", (lambda x: x+1, )),  # add in when pandas 2.1 is out
    ],
)
@pytest.mark.parametrize("indexer", ["x", ["x"]])
def test_simplify_up_blockwise(df, pdf, func, args, indexer):
    q = getattr(df, func)(*args)[indexer]
    result = q.simplify()
    expected = getattr(df[indexer], func)(*args)
    assert result._name == expected._name

    assert_eq(q, getattr(pdf, func)(*args)[indexer])

    q = getattr(df, func)(*args)[["x", "y"]]
    result = q.simplify()
    expected = getattr(df, func)(*args)
    assert result._name == expected._name


def test_align(df, pdf):
    result_1, result_2 = df.align(df)
    pdf_result_1, pdf_result_2 = pdf.align(pdf)
    assert_eq(result_1, pdf_result_1)
    assert_eq(result_2, pdf_result_2)

    result_1, result_2 = df.x.align(df.x)
    pdf_result_1, pdf_result_2 = pdf.x.align(pdf.x)
    assert_eq(result_1, pdf_result_1)
    assert_eq(result_2, pdf_result_2)


def test_assign_simplify(pdf):
    df = DataFrame(pdf)
    df2 = DataFrame(pdf)
    df["new"] = df.x > 1
    result = df[["x", "new"]].simplify()
    expected = df2[["x"]].assign(new=df2.x > 1).simplify()
    assert result._name == expected._name

    pdf["new"] = pdf.x > 1
    assert_eq(pdf[["x", "new"]], result)


def test_assign_simplify_new_column_not_needed(pdf):
    df = DataFrame(pdf)
    df2 = DataFrame(pdf)
    df["new"] = df.x > 1
    result = df[["x"]].simplify()
    expected = df2[["x"]].simplify()
    assert result._name == expected._name

    pdf["new"] = pdf.x > 1
    assert_eq(result, pdf[["x"]])


def test_assign_simplify_series(pdf):
    df = DataFrame(pdf)
    df2 = DataFrame(pdf)
    df["new"] = df.x > 1
    result = df.new.simplify()
    expected = df2[[]].assign(new=df2.x > 1).new.simplify()
    assert result._name == expected._name


def test_assign_non_series_inputs(df, pdf):
    assert_eq(df.assign(a=lambda x: x.x * 2), pdf.assign(a=lambda x: x.x * 2))
    assert_eq(df.assign(a=2), pdf.assign(a=2))
    assert_eq(df.assign(a=df.x.sum()), pdf.assign(a=pdf.x.sum()))

    assert_eq(df.assign(a=lambda x: x.x * 2).y, pdf.assign(a=lambda x: x.x * 2).y)
    assert_eq(df.assign(a=lambda x: x.x * 2).a, pdf.assign(a=lambda x: x.x * 2).a)


def test_astype_categories(df):
    result = df.astype("category")
    assert_eq(result.x._meta.cat.categories, pd.Index([UNKNOWN_CATEGORIES]))
    assert_eq(result.y._meta.cat.categories, pd.Index([UNKNOWN_CATEGORIES]))


def test_drop_simplify(pdf, df):
    q = df.drop(columns=["x"])[["y"]]
    result = q.simplify()
    expected = df[["y"]]
    assert result._name == expected._name


def test_op_align():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": 1})
    df = DataFrame(pdf)

    pdf2 = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": 1})
    df2 = DataFrame(pdf2)

    assert_eq(df - df2, pdf - pdf2)
