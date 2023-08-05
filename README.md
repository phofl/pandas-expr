# pandas Expressions

Lazy pandas API POC.

## Reading from parquet files

Prepare the parquet file:

```python
import pandas as pd

pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": 1, "d": 1.5}).to_parquet("test.parquet")
```

```python
from pandas_expr import read_parquet

df = read_parquet("test.parquet")
result = df[df["b"] == "x"][["a", "c"]]
```

Let's look at how this query looks:

```python
result.pprint()

Projection: columns=['a', 'c']
  Filter:
    ReadParquet: path='test.parquet'
    EQ: right='x'
      Projection: columns='b'
        ReadParquet: path='test.parquet'
```

No need to read all of the data, we can do better:

```python
result.optimize().pprint()

ReadParquet: path='test.parquet' columns=['a', 'c'] filters=[[('b', '==', 'x')]]
```

We pushed the column selection and the filter into the ``read_parquet`` call.


## DataFrame constructor

The DataFrame constructor mirrors the regular pandas constructor, but it is
lazy and does not trigger any actual computation.

```python
from pandas_expr import DataFrame

df = DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": 1, "d": 1.5})
df = df.replace(1, 5).fillna(100)[["a", "b"]]

df.pprint()

Projection: columns=['a', 'b']
  Fillna: value=100
    Replace: to_replace=1 value=5
      PandasIO: data={'a': [1, 2, 3], 'b': ['x', 'y', 'z'], 'c': 1, 'd': 1.5}
```

We can again make this more efficient:

```python
df.optimize(fuse=False).pprint()

Fillna: value=100
  Replace: to_replace=1 value=5
    Projection: columns=['a', 'b']
      PandasIO: data={'a': [1, 2, 3], 'b': ['x', 'y', 'z'], 'c': 1, 'd': 1.5}
```
