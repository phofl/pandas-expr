from pandas_expr.io.io import BlockwiseIO


class ReadCSV(BlockwiseIO):
    _parameters = ["filename", "usecols", "header", "_partitions", "storage_options"]
    _defaults = {
        "usecols": None,
        "header": "infer",
        "_partitions": None,
        "storage_options": None,
    }

    @property
    def _meta(self):
        return self._ddf._meta
