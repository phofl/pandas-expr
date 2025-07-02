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
        from pandas_expr import Series

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
        from pandas_expr._collection import new_collection

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


class StringAccessor(Accessor):
    """String accessor for Series operations"""
    _accessor_name = "str"
    
    # Common string methods that should be available
    _accessor_methods = [
        "upper", "lower", "title", "capitalize", "swapcase",
        "strip", "lstrip", "rstrip", "replace", "contains",
        "startswith", "endswith", "split", "rsplit", "slice",
        "extract", "extractall", "findall", "match", "count",
        "pad", "center", "ljust", "rjust", "zfill", "wrap",
        "encode", "decode", "normalize", "translate", "len",
    ]
    
    _accessor_properties = []


class DatetimeAccessor(Accessor):
    """Datetime accessor for Series operations"""
    _accessor_name = "dt"
    
    # Common datetime methods/properties
    _accessor_methods = [
        "strftime", "round", "floor", "ceil", "month_name", 
        "day_name", "normalize", "to_period", "to_pydatetime",
        "tz_localize", "tz_convert",
    ]
    
    _accessor_properties = []
    
    # Manually implement datetime properties that return Series
    @property
    def year(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "year"))
    
    @property
    def month(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "month"))
    
    @property
    def day(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "day"))
    
    @property
    def hour(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "hour"))
    
    @property
    def minute(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "minute"))
    
    @property
    def second(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "second"))
    
    @property
    def microsecond(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "microsecond"))
    
    @property
    def nanosecond(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "nanosecond"))
    
    @property
    def date(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "date"))
    
    @property
    def time(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "time"))
    
    @property
    def dayofyear(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "dayofyear"))
    
    @property
    def weekofyear(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "weekofyear"))
    
    @property
    def week(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "week"))
    
    @property
    def dayofweek(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "dayofweek"))
    
    @property
    def weekday(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "weekday"))
    
    @property
    def quarter(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "quarter"))
    
    # These are scalar properties - delegate to metadata
    @property
    def freq(self):
        return self._delegate_property(self._series._meta, "dt", "freq")
    
    @property
    def tz(self):
        return self._delegate_property(self._series._meta, "dt", "tz")
    
    # Boolean properties that return Series
    @property
    def is_month_start(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "is_month_start"))
    
    @property
    def is_month_end(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "is_month_end"))
    
    @property
    def is_quarter_start(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "is_quarter_start"))
    
    @property
    def is_quarter_end(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "is_quarter_end"))
    
    @property
    def is_year_start(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "is_year_start"))
    
    @property
    def is_year_end(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "is_year_end"))
    
    @property
    def is_leap_year(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "is_leap_year"))
    
    @property
    def days_in_month(self):
        from pandas_expr._collection import new_collection
        return new_collection(PropertyMap(self._series.expr, "dt", "days_in_month"))
