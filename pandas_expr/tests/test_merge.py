import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq

from pandas_expr import DataFrame


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
@pytest.mark.parametrize("shuffle_backend", ["tasks", "disk"])
def test_merge(how, shuffle_backend):
    # Make simple left & right dfs
    pdf1 = pd.DataFrame({"x": range(20), "y": range(20)})
    df1 = DataFrame(pdf1)
    pdf2 = pd.DataFrame({"x": range(0, 20, 2), "z": range(10)})
    df2 = DataFrame(pdf2)

    # Partition-wise merge with map_partitions
    df3 = df1.merge(df2, on="x", how=how)

    # Check result with/without fusion
    expect = pdf1.merge(pdf2, on="x", how=how)
    assert_eq(df3, expect, check_index=False)
    assert_eq(df3.optimize(fuse=False), expect, check_index=False)


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
@pytest.mark.parametrize("pass_name", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("shuffle_backend", ["tasks", "disk"])
def test_merge_indexed(how, pass_name, sort, shuffle_backend):
    # Make simple left & right dfs
    pdf1 = pd.DataFrame({"x": range(20), "y": range(20)}).set_index("x")
    df1 = DataFrame(pdf1)
    pdf2 = pd.DataFrame({"x": range(0, 20, 2), "z": range(10)}).set_index("x")
    df2 = DataFrame(pdf2)

    if pass_name:
        left_on = right_on = "x"
        left_index = right_index = False
    else:
        left_on = right_on = None
        left_index = right_index = True

    df3 = df1.merge(
        df2,
        left_index=left_index,
        left_on=left_on,
        right_index=right_index,
        right_on=right_on,
        how=how,
    )

    # Check result with/without fusion
    expect = pdf1.merge(
        pdf2,
        left_index=left_index,
        left_on=left_on,
        right_index=right_index,
        right_on=right_on,
        how=how,
    )
    assert_eq(df3, expect)
    assert_eq(df3.optimize(fuse=False), expect)


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
def test_broadcast_merge(how):
    # Make simple left & right dfs
    pdf1 = pd.DataFrame({"x": range(20), "y": range(20)})
    df1 = DataFrame(pdf1)
    pdf2 = pd.DataFrame({"x": range(0, 20, 2), "z": range(10)})
    df2 = DataFrame(pdf2)

    df3 = df1.merge(df2, on="x", how=how)

    # Check that we avoid the shuffle when allowed
    if how in ("left", "inner"):
        assert all(["Shuffle" not in str(op) for op in df3.simplify().operands[:2]])

    # Check result with/without fusion
    expect = pdf1.merge(pdf2, on="x", how=how)
    assert_eq(df3, expect, check_index=False)
    assert_eq(df3.optimize(fuse=False), expect, check_index=False)


@pytest.mark.xfail(reason="Projection is missing")
def test_merge_column_projection():
    # Make simple left & right dfs
    pdf1 = pd.DataFrame({"x": range(20), "y": range(20), "z": range(20)})
    df1 = DataFrame(pdf1)
    pdf2 = pd.DataFrame({"x": range(0, 20, 2), "z": range(10)})
    df2 = DataFrame(pdf2)

    # Partition-wise merge with map_partitions
    df3 = df1.merge(df2, on="x")["z_x"].simplify()

    assert "y" not in df3.expr.operands[0].columns


@pytest.mark.parametrize("how", ["left", "right", "inner", "outer"])
@pytest.mark.parametrize("shuffle_backend", ["tasks", "disk"])
def test_join(how, shuffle_backend):
    # Make simple left & right dfs
    pdf1 = pd.DataFrame({"x": range(20), "y": range(20)})
    df1 = DataFrame(pdf1)
    pdf2 = pd.DataFrame({"z": range(10)}, index=pd.Index(range(10), name="a"))
    df2 = DataFrame(pdf2)

    # Partition-wise merge with map_partitions
    df3 = df1.join(df2, on="x", how=how)

    # Check result with/without fusion
    expect = pdf1.join(pdf2, on="x", how=how)
    assert_eq(df3, expect, check_index=False)
    assert_eq(df3.optimize(fuse=False), expect, check_index=False)

    df3 = df1.join(df2.z, on="x", how=how)
    assert_eq(df3, expect, check_index=False)
    assert_eq(df3.optimize(fuse=False), expect, check_index=False)


def test_join_recursive():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": 1}, index=pd.Index([1, 2, 3], name="a"))
    df = DataFrame(pdf)

    pdf2 = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": 1}, index=pd.Index([1, 2, 3, 4, 5, 6], name="a")
    )
    df2 = DataFrame(pdf2)

    pdf3 = pd.DataFrame({"c": [1, 2, 3], "d": 1}, index=pd.Index([1, 2, 3], name="a"))
    df3 = DataFrame(pdf3)

    result = df.join([df2, df3], how="outer")
    assert_eq(result, pdf.join([pdf2, pdf3], how="outer"))

    result = df.join([df2, df3], how="left")
    # The nature of our join might cast ints to floats
    assert_eq(result, pdf.join([pdf2, pdf3], how="left"), check_dtype=False)
