import functools

import pandas as pd

from pandas_expr.io.io import BlockwiseIO


class ReadCSV(BlockwiseIO):
    _parameters = ["filename", "usecols", "header", "storage_options"]
    _defaults = {
        "usecols": None,
        "header": "infer",
        "storage_options": None,
    }
    _keyword_only = ["usecols", "header", "storage_options"]
    operation = staticmethod(pd.read_csv)

    @functools.cached_property
    def _meta(self):
        return pd.read_csv(self.filename, **self._kwargs, nrows=1).iloc[:0]
