from dask.base import normalize_token, tokenize  # noqa: F401
from dask.dataframe.utils import make_meta, meta_nonempty  # noqa: F401
from dask.dataframe.utils import (  # noqa: F401
    clear_known_categories,
    strip_unknown_categories,
)


def collections_to_dsk(collections, optimize_graph=True, **kwargs):
    """Convert collections to a dask graph"""
    import toolz
    graphs = []
    for collection in collections:
        graphs.append(collection.__dask_graph__())
    return toolz.merge(*graphs)
