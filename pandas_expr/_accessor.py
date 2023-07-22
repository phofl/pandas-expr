from dask.dataframe.accessor import _bind_method, _bind_property, maybe_wrap_pandas

from pandas_expr._expr import Elemwise


class Accessor:
    """
    Base class for pandas Accessor objects cat, dt, and str.

    Notes
    -----
    Subclasses should define ``_accessor_name``, ``_accessor_methods``, and
    ``_accessor_properties``.
    """

    def __init__(self, series):
        from dask_expr import Series

        if not isinstance(series, Series):
            raise ValueError("Accessor cannot be initialized")

        series_meta = series._meta
        if hasattr(series_meta, "to_series"):  # is index-like
            series_meta = series_meta.to_series()
        meta = getattr(series_meta, self._accessor_name)

        self._meta = meta
        self._series = series

    def __init_subclass__(cls, **kwargs):
        """Bind all auto-generated methods & properties"""
        import pandas as pd

        super().__init_subclass__(**kwargs)
        pd_cls = getattr(pd.Series, cls._accessor_name)
        for item in cls._accessor_methods:
            attr, min_version = item if isinstance(item, tuple) else (item, None)
            if not hasattr(cls, attr):
                _bind_method(cls, pd_cls, attr, min_version)
        for item in cls._accessor_properties:
            attr, min_version = item if isinstance(item, tuple) else (item, None)
            if not hasattr(cls, attr):
                _bind_property(cls, pd_cls, attr, min_version)

    @staticmethod
    def _delegate_property(obj, accessor, attr):
        out = getattr(getattr(obj, accessor, obj), attr)
        return maybe_wrap_pandas(obj, out)

    @staticmethod
    def _delegate_method(obj, accessor, attr, args, kwargs):
        out = getattr(getattr(obj, accessor, obj), attr)(*args, **kwargs)
        return maybe_wrap_pandas(obj, out)

    def _function_map(self, attr, *args, **kwargs):
        from dask_expr._collection import new_collection

        return new_collection(
            FunctionMap(self._series.expr, self._accessor_name, attr, args, kwargs)
        )


class PropertyMap(Elemwise):
    _parameters = [
        "frame",
        "accessor",
        "attr",
    ]

    def operation(self, obj, accessor, attr):
        out = getattr(getattr(obj, accessor, obj), attr)
        return maybe_wrap_pandas(obj, out)


class FunctionMap(Elemwise):
    _parameters = ["frame", "accessor", "attr", "args", "kwargs"]

    def operation(self, obj, accessor, attr, args, kwargs):
        out = getattr(getattr(obj, accessor, obj), attr)(*args, **kwargs)
        return maybe_wrap_pandas(obj, out)
