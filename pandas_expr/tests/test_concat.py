import numpy as np
import pandas as pd
import pytest
from dask.dataframe import assert_eq

from pandas_expr import DataFrame, concat


@pytest.fixture
def pdf():
    pdf = pd.DataFrame({"x": range(100)})
    pdf["y"] = pdf.x * 10.0
    yield pdf


@pytest.fixture
def df(pdf):
    yield DataFrame(pdf)


def test_concat(pdf, df):
    result = concat([df, df])
    expected = pd.concat([pdf, pdf])
    assert_eq(result, expected)


def test_concat_pdf(pdf, df):
    result = concat([df, pdf])
    expected = pd.concat([pdf, pdf])
    assert_eq(result, expected)


def test_concat_divisions(pdf, df):
    pdf2 = pdf.set_index(np.arange(200, 300))
    df2 = DataFrame(pdf2)
    result = concat([df, df2])
    expected = pd.concat([pdf, pdf2])
    assert_eq(result, expected)


@pytest.mark.parametrize("join", ["right", "left"])
def test_invalid_joins(join):
    with pytest.raises(ValueError, match="Only can inner"):
        concat([df, df], join=join)


def test_concat_one_object(df, pdf):
    result = concat([df])
    expected = pd.concat([pdf])
    assert_eq(result, expected)


def test_concat_one_no_columns(df, pdf):
    result = concat([df, df[[]]])
    expected = pd.concat([pdf, pdf[[]]])
    assert_eq(result, expected)


def test_concat_simplify(pdf, df):
    pdf2 = pdf.copy()
    pdf2["z"] = 1
    df2 = DataFrame(pdf2)
    q = concat([df, df2])[["z", "x"]]
    result = q.simplify()
    expected = concat([df[["x"]], df2[["x", "z"]]])[["z", "x"]]
    assert result._name == expected._name

    assert_eq(q, pd.concat([pdf, pdf2])[["z", "x"]])


def test_concat_simplify_projection_not_added(pdf, df):
    pdf2 = pdf.copy()
    pdf2["z"] = 1
    df2 = DataFrame(pdf2)
    q = concat([df, df2])[["y", "x"]]
    result = q.simplify()
    expected = concat([df, df2[["x", "y"]]]).simplify()[["y", "x"]]
    assert result._name == expected._name

    assert_eq(q, pd.concat([pdf, pdf2])[["y", "x"]])
