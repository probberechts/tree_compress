import veritas
from .compress import Compress

def compress_topdown(data, clf, relerr=0.01, max_rounds=2, return_compress_object=False, silent=False):
    if isinstance(clf, veritas.AddTree):
        at = clf
    else:
        at = veritas.get_addtree(clf)

    compress = Compress(data, at, silent=silent)
    at_pruned = compress.compress(relerr=relerr, max_rounds=max_rounds)

    if return_compress_object:
        return compress
    else:
        return at_pruned


def refine_leafs(data, clf):
    pass


def prune_ensemble(data, clf):
    pass

