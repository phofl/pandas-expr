import os

import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq

from pandas_expr import optimize, read_parquet
from pandas_expr._expr import Replace
from pandas_expr._reductions import Len
from pandas_expr.io import ReadParquet


def _make_file(dir, format="parquet", df=None):
    fn = os.path.join(str(dir), f"myfile.{format}")
    if df is None:
        df = pd.DataFrame({c: range(10) for c in "abcde"})
    if format == "csv":
        df.to_csv(fn)
    elif format == "parquet":
        df.to_parquet(fn)
    else:
        ValueError(f"{format} not a supported format")
    return fn


def df(fn):
    return read_parquet(fn, columns=["a", "b", "c"])


def df_bc(fn):
    return read_parquet(fn, columns=["b", "c"])


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            # Add -> Mul
            lambda fn: df(fn) + df(fn),
            lambda fn: 2 * df(fn),
        ),
        (
            # Column projection
            lambda fn: df(fn)[["b", "c"]],
            lambda fn: read_parquet(fn, columns=["b", "c"]),
        ),
        (
            # Compound
            lambda fn: 3 * (df(fn) + df(fn))[["b", "c"]],
            lambda fn: 6 * df_bc(fn),
        ),
        (
            # Traverse Sum
            lambda fn: df(fn).sum()[["b", "c"]],
            lambda fn: df_bc(fn).sum(),
        ),
        (
            # Respect Sum keywords
            lambda fn: df(fn).sum(numeric_only=True)[["b", "c"]],
            lambda fn: df_bc(fn).sum(numeric_only=True),
        ),
    ],
)
def test_optimize(tmpdir, input, expected):
    fn = _make_file(tmpdir, format="parquet")
    result = optimize(input(fn), fuse=False)
    assert str(result.expr) == str(expected(fn).expr)


def test_predicate_pushdown(tmpdir):
    original = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5] * 10,
            "b": [0, 1, 2, 3, 4] * 10,
            "c": range(50),
            "d": [6, 7] * 25,
            "e": [8, 9] * 25,
        }
    )
    fn = _make_file(tmpdir, format="parquet", df=original)
    df = read_parquet(fn)
    assert_eq(df, original)
    x = df[df.a == 5][df.c > 20]["b"]
    y = optimize(x, fuse=False)
    assert isinstance(y.expr, ReadParquet)
    assert ("a", "==", 5) in y.expr.operand("filters")[0]
    assert ("c", ">", 20) in y.expr.operand("filters")[0]
    assert list(y.columns) == ["b"]

    # Check computed result
    y_result = y.compute()
    assert y_result.name == "b"
    assert len(y_result) == 6
    assert all(y_result == 4)


def test_predicate_pushdown_compound(tmpdir):
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5] * 10,
            "b": [0, 1, 2, 3, 4] * 10,
            "c": range(50),
            "d": [6, 7] * 25,
            "e": [8, 9] * 25,
        }
    )
    fn = _make_file(tmpdir, format="parquet", df=pdf)
    df = read_parquet(fn)

    # Test AND
    x = df[(df.a == 5) & (df.c > 20)]["b"]
    y = optimize(x, fuse=False)
    assert isinstance(y.expr, ReadParquet)
    assert {("c", ">", 20), ("a", "==", 5)} == set(y.filters[0])
    assert_eq(
        y,
        pdf[(pdf.a == 5) & (pdf.c > 20)]["b"],
        check_index=False,
    )

    # Test OR
    x = df[(df.a == 5) | (df.c > 20)][df.b != 0]["b"]
    y = optimize(x, fuse=False)
    assert isinstance(y.expr, ReadParquet)
    filters = [set(y.filters[0]), set(y.filters[1])]
    assert {("c", ">", 20), ("b", "!=", 0)} in filters
    assert {("a", "==", 5), ("b", "!=", 0)} in filters
    assert_eq(
        y,
        pdf[(pdf.a == 5) | (pdf.c > 20)][pdf.b != 0]["b"],
        check_index=False,
    )

    # Test OR and AND
    x = df[((df.a == 5) | (df.c > 20)) & (df.b != 0)]["b"]
    z = optimize(x, fuse=False)
    assert isinstance(z.expr, ReadParquet)
    filters = [set(z.filters[0]), set(z.filters[1])]
    assert {("c", ">", 20), ("b", "!=", 0)} in filters
    assert {("a", "==", 5), ("b", "!=", 0)} in filters
    assert_eq(y, z)


def test_parquet_complex_filters(tmpdir):
    df = read_parquet(_make_file(tmpdir))
    pdf = df.compute()
    got = df["a"][df["b"] > df["b"].mean()]
    expect = pdf["a"][pdf["b"] > pdf["b"].mean()]

    assert_eq(got, expect)
    assert_eq(got.optimize(fuse=False), expect)


def test_parquet_len_filter(tmpdir):
    df = read_parquet(_make_file(tmpdir))
    expr = Len(df[df.c > 0].expr)
    result = expr.simplify()
    for rp in result.find_operations(ReadParquet):
        assert rp.operand("columns") == ["c"] or rp.operand("columns") == []


# def test_to_parquet(tmpdir):
#     pdf = pd.DataFrame({"x": [1, 4, 3, 2, 0, 5]})
#     df = DataFrame(pdf)
#
#     # Check basic parquet round trip
#     df.to_parquet(tmpdir + "/test.parquet")
#     df2 = read_parquet(tmpdir + "/test.parquet")
#     assert_eq(df, df2)
#
#     # Check overwrite behavior
#     df["new"] = df["x"] + 1
#     df.to_parquet(tmpdir + "/test.parquet")
#     df2 = read_parquet(tmpdir + "/test.parquet")
#     assert_eq(df, df2)


def test_combine_similar(tmpdir):
    pdf = pd.DataFrame(
        {"x": [0, 1, 2, 3] * 4, "y": range(16), "z": [None, 1, 2, 3] * 4}
    )
    fn = _make_file(tmpdir, format="parquet", df=pdf)
    df = read_parquet(fn)
    df = df.replace(1, 100)
    df["xx"] = df.x != 0
    df["yy"] = df.y != 0
    got = df[["xx", "yy", "x"]].sum()

    pdf = pdf.replace(1, 100)
    pdf["xx"] = pdf.x != 0
    pdf["yy"] = pdf.y != 0
    expect = pdf[["xx", "yy", "x"]].sum()

    # Check correctness
    assert_eq(got, expect)
    assert_eq(got.optimize(fuse=False), expect)

    # We should only have one ReadParquet node, and
    # it should not include "z" in the column projection
    read_parquet_nodes = list(got.optimize(fuse=False).find_operations(ReadParquet))
    assert len(read_parquet_nodes) == 1
    assert set(read_parquet_nodes[0].columns) == {"x", "y"}

    # All Replace operations should also be the same
    replace_nodes = list(got.optimize(fuse=False).find_operations(Replace))
    assert len(replace_nodes) == 1
