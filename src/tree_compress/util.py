import numpy as np
from veritas import AddTreeType

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import accuracy_score


def neg_root_mean_squared_error(ytrue, ypred):
    return -root_mean_squared_error(ytrue, ypred)


def count_nnz_leafs(at):
    nnz = 0
    for t in at:
        for lid in t.get_leaf_ids():
            lvals = t.get_leaf_values(lid)
            nnz += int(np.any(np.abs(lvals) > 1e-5))
    return nnz


def metric(at, ytrue, x=None, ypred=None):
    at_type = at.get_type()
    nlv = at.num_leaf_values()

    if at_type in {AddTreeType.REGR, AddTreeType.REGR_MEAN}:
        score = neg_root_mean_squared_error
        if ypred is None:
            ypred = at.predict(x)
    elif at_type in {AddTreeType.CLF_SOFTMAX, AddTreeType.CLF_MEAN}:
        score = accuracy_score
        if ypred is None and nlv == 1:
            ypred = at.predict(x) > 0.5
        elif ypred is None:
            ypred = np.argmax(at.predict(x), axis=1)
    else:
        raise RuntimeError("cannot determine task")

    return score(ytrue, ypred)


def metric_name(at):
    at_type = at.get_type()
    if at_type in {AddTreeType.REGR, AddTreeType.REGR_MEAN}:
        return "neg_rmse"
    elif at_type in {AddTreeType.CLF_SOFTMAX, AddTreeType.CLF_MEAN}:
        return "accuracy"
    else:
        raise RuntimeError("cannot determine task")


def isworse_relerr(metric, reference, relerr=0.0):  # higher is better
    eps = (metric - reference) / abs(reference)
    return eps <= -relerr


def is_almost_eq(metric, reference, relerr=1e-5):
    eps = abs((metric - reference) / reference)
    return eps < relerr


def is_not_almost_eq(metric, reference, relerr=1e-5):
    return not is_almost_eq(metric, reference, relerr)


def print_metrics(prefix, r, rcmp=None, cmp=isworse_relerr):
    import colorama

    RST = colorama.Style.RESET_ALL
    RED = colorama.Fore.RED
    GRN = colorama.Fore.GREEN
    BLD = colorama.Style.BRIGHT

    if rcmp is not None:
        ctr = RED if cmp(r.mtrain, rcmp.mtrain) else GRN
        cte = RED if cmp(r.mtest, rcmp.mtest) else GRN
        cva = RED if cmp(r.mvalid, rcmp.mvalid) else GRN
    else:
        ctr, cte, cva = "", "", ""

    print(
        f"METRICS {prefix:6s}",
        f"{ctr}tr {r.mtrain:.3f}{RST},",
        f"{cva}va {r.mvalid:.3f}{RST}",
        f"{BLD}{cte}[te {r.mtest:.3f}]{RST},",
        f" ntrees {r.ntrees:3d},",
        f" nnodes {r.nnodes:5d},",
        f" nleafs {r.nleafs:5d},",
        f" nnz {r.nnz_leafs:5d}",
    )


def print_fit(r, alpha_search):
    import colorama

    RST = colorama.Style.RESET_ALL

    mtrain = alpha_search.compress.mtrain
    mvalid = alpha_search.compress.mvalid

    ctr = _color(r.mtrain_clf, mtrain, alpha_search)
    cte = _color(r.mvalid_clf, mvalid, alpha_search)

    status = f"{colorama.Fore.GREEN}fit ok{RST}"
    if alpha_search.underfits(r.mtrain_clf, r.mvalid_clf):
        status = f"{colorama.Fore.YELLOW}under {RST}"
    elif alpha_search.overfits(r.mtrain_clf, r.mvalid_clf):
        status = f"{colorama.Fore.RED}over  {RST}"

    print(f"{mtrain:7.3f} {mvalid:7.3f} ->", end=" ")

    ndigits = int(np.ceil(np.log10(1 + r.num_params)))

    print(
        f"{ctr}{r.mtrain_clf:7.3f}{RST} {cte}{r.mvalid_clf:7.3f}{RST},",
        f"{r.frac_removed*100:3.0f}% removed",
        f"(alpha={r.alpha:9.4f},",
        # f"nnz={r['num_kept']}/{r['num_params']})",
        "nnz={0:{n}d}/{1:{n}d})".format(r.num_kept, r.num_params, n=ndigits),
        status,
        np.power(10.0, [alpha_search.lo, alpha_search.hi]).round(4),
    )


def _color(metric, reference, alpha_search):
    from colorama import Fore

    if not alpha_search.isworse_fun(metric, reference):
        return Fore.GREEN
    return Fore.RED
