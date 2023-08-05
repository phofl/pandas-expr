import functools

import pandas as pd

from pandas_expr._deps import make_meta, meta_nonempty, strip_unknown_categories
from pandas_expr._expr import Blockwise, Projection


class Concat(Blockwise):
    _parameters = ["join"]
    _defaults = {"join": "outer"}
    _keyword_only = ["join"]

    def operation(self, *args, **kwargs):
        return pd.concat(args, **kwargs)

    def __str__(self):
        s = (
            "frames="
            + str(self.dependencies())
            + ", "
            + ", ".join(
                str(param) + "=" + str(operand)
                for param, operand in zip(self._parameters, self.operands)
                if operand != self._defaults.get(param)
            )
        )
        return f"{type(self).__name__}({s})"

    @property
    def _frames(self):
        return self.dependencies()

    @functools.cached_property
    def _meta(self):
        meta = make_meta(
            pd.concat(
                [meta_nonempty(df._meta) for df in self._frames],
                **self._kwargs,
            )
        )
        return strip_unknown_categories(meta)

    def _simplify_up(self, parent):
        if isinstance(parent, Projection):
            columns = parent.columns
            columns_frame = [
                sorted(set(frame.columns).intersection(columns))
                for frame in self._frames
            ]
            if all(
                cols == sorted(frame.columns)
                for frame, cols in zip(self._frames, columns_frame)
            ):
                return

            frames = [
                frame[cols] if cols != sorted(frame.columns) else frame
                for frame, cols in zip(self._frames, columns_frame)
            ]
            return type(parent)(
                type(self)(self.join, *frames),
                *parent.operands[1:],
            )
