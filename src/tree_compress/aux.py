from functools import partial

import veritas

from .lasso_compress import Compress
from .util import default_metric, isworse_abserr, isworse_relerr


def compress_topdown(
    data,
    clf,
    relerr=0.01,
    abserr=None,
    max_rounds=2,
    return_compress_object=False,
    silent=False,
):
    at = clf if isinstance(clf, veritas.AddTree) else veritas.get_addtree(clf)

    metric_name, metric = default_metric(at)
    print(metric_name)

    if abserr is not None and relerr is not None:
        raise ValueError("Only one of abserr or relerr can be set.")
    elif abserr is not None:
        isworse = partial(isworse_abserr, abserr=abserr)
    elif relerr is not None:
        isworse = partial(isworse_relerr, relerr=relerr)
    else:
        raise ValueError("Either abserr or relerr must be set.")

    compress = Compress(data, at, score=metric, isworse=isworse, silent=silent)
    at_pruned = compress.compress(max_rounds=max_rounds)

    return compress if return_compress_object else at_pruned


def refine_leafs(data, clf):
    raise NotImplementedError("Not implemented yet.")


def prune_ensemble(data, clf):
    raise NotImplementedError("Not implemented yet.")
