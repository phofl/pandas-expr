import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq

from pandas_expr import DataFrame


@pytest.fixture
def pdf():
    pdf = pd.DataFrame({"x": list(range(10)) * 10, "y": range(100), "z": 1})
    yield pdf


@pytest.fixture
def df(pdf):
    yield DataFrame(pdf)


@pytest.mark.xfail(reason="Cannot group on a Series yet")
def test_groupby_unsupported_by(pdf, df):
    assert_eq(df.groupby(df.x).sum(), pdf.groupby(pdf.x).sum())


@pytest.mark.parametrize(
    "api", ["sum", "mean", "min", "max", "prod", "first", "last", "var", "std"]
)
@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_numeric(pdf, df, api, numeric_only):
    g = df.groupby("x")
    agg = getattr(g, api)(numeric_only=numeric_only)

    expect = getattr(pdf.groupby("x"), api)(numeric_only=numeric_only)
    assert_eq(agg, expect)

    g = df.groupby("x")
    agg = getattr(g, api)(numeric_only=numeric_only)["y"]

    expect = getattr(pdf.groupby("x"), api)(numeric_only=numeric_only)["y"]
    assert_eq(agg, expect)


@pytest.mark.parametrize("func", ["count", "value_counts", "size"])
def test_groupby_no_numeric_only(pdf, func):
    pdf = pdf.drop(columns="z")
    df = DataFrame(pdf)
    g = df.groupby("x")
    agg = getattr(g, func)()

    expect = getattr(pdf.groupby("x"), func)()
    assert_eq(agg, expect)


def test_groupby_mean_slice(pdf, df):
    g = df.groupby("x")
    agg = g.y.mean()

    expect = pdf.groupby("x").y.mean()
    assert_eq(agg, expect)


def test_groupby_series(pdf, df):
    pdf_result = pdf.groupby(pdf.x).sum()
    result = df.groupby(df.x).sum()
    assert_eq(result, pdf_result)
    result = df.groupby("x").sum()
    assert_eq(result, pdf_result)

    df2 = DataFrame(pd.DataFrame({"a": [1, 2, 3]}))

    with pytest.raises(ValueError, match="DataFrames columns"):
        df.groupby(df2.a)


@pytest.mark.parametrize(
    "spec",
    [
        {"x": "count"},
        {"x": ["count"]},
        {"x": ["count"], "y": "mean"},
        {"x": ["sum", "mean"]},
        ["min", "mean"],
        "sum",
    ],
)
def test_groupby_agg(pdf, df, spec):
    g = df.groupby("x")
    agg = g.agg(spec)

    expect = pdf.groupby("x").agg(spec)
    assert_eq(agg, expect)


def test_groupby_getitem_agg(pdf, df):
    assert_eq(df.groupby("x").y.sum(), pdf.groupby("x").y.sum())
    assert_eq(df.groupby("x")[["y"]].sum(), pdf.groupby("x")[["y"]].sum())


def test_groupby_agg_column_projection(pdf, df):
    g = df.groupby("x")
    agg = g.agg({"x": "count"}).simplify()

    assert list(agg.frame.columns) == ["x"]
    expect = pdf.groupby("x").agg({"x": "count"})
    assert_eq(agg, expect)
