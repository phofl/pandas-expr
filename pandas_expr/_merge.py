import functools

from pandas_expr._expr import Blockwise, Expr
from pandas_expr._util import M


class Merge(Blockwise):
    _parameters = [
        "left",
        "right",
        "how",
        "on",
        "left_on",
        "right_on",
        "left_index",
        "right_index",
        "suffixes",
        "indicator",
    ]
    _defaults = {
        "how": "inner",
        "on": None,
        "left_on": None,
        "right_on": None,
        "left_index": False,
        "right_index": False,
        "suffixes": ("_x", "_y"),
        "indicator": False,
    }
    _keyword_only = [
        "how",
        "on",
        "left_on",
        "right_on",
        "left_index",
        "right_index",
        "suffixes",
        "indicator",
    ]
    operation = M.merge


class JoinRecursive(Expr):
    _parameters = ["frames", "how"]
    _defaults = {"right_index": True, "how": "outer"}

    @functools.cached_property
    def _meta(self):
        if len(self.frames) == 1:
            return self.frames[0]._meta
        else:
            return self.frames[0]._meta.join(
                [op._meta for op in self.frames[1:]],
            )

    def _lower(self):
        if self.how == "left":
            right = self._recursive_join(self.frames[1:])
            return Merge(
                self.frames[0],
                right,
                how=self.how,
                left_index=True,
                right_index=True,
            )

        return self._recursive_join(self.frames)

    def _recursive_join(self, frames):
        if len(frames) == 1:
            return frames[0]

        if len(frames) == 2:
            return Merge(
                frames[0],
                frames[1],
                how="outer",
                left_index=True,
                right_index=True,
            )

        midx = len(self.frames) // 2

        return self._recursive_join(
            [
                self._recursive_join(frames[:midx]),
                self._recursive_join(frames[midx:]),
            ],
        )
