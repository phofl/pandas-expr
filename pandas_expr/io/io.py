from __future__ import annotations

import pandas as pd

from pandas_expr._expr import Blockwise, Expr


class IO(Expr):
    def __str__(self):
        return f"{type(self).__name__}({self._name[-7:]})"


class BlockwiseIO(Blockwise, IO):
    pass


class PandasIO(Blockwise, IO):
    _parameters = ["data", "index", "columns", "dtype", "copy"]
    _defaults = {"index": None, "columns": None, "dtype": None, "copy": None}
    operation = pd.DataFrame
