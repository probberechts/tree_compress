from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from veritas import AddTreeType


@dataclass
class Data:
    """The train, test, and validation data and labels.

    Attributes:
        xtrain (np.ndarray): The data that was used to train the tree ensemble.
        ytrain (np.ndarray): Train labels.

        xtest  (np.ndarray): The data that is used to evaluate the tree ensemble and the
            pruned tree ensemble. This data is **not** used in the learning or the
            compression process.
        ytest  (np.ndarray): Test labels.

        xvalid (np.ndarray): This data is used to tune the strenght of the
            regularization coefficient alpha. This data should not have been used to
            train the ensemble to avoid overfitting on the data used to train the
            ensemble.
        yvalid (np.ndarray): Validation labels.

    """

    # The data that was used to train the tree ensemble
    xtrain: np.ndarray
    ytrain: np.ndarray

    # The data that is used to evaluate the tree ensemble and the pruned tree ensemble.
    # This data is **not** used in the learning or the compression process.
    xtest: np.ndarray
    ytest: np.ndarray

    # This data is used to tune the strenght of the regularization coefficient alpha.
    # This data should not have been used to train the ensemble to avoid overfitting on
    # the data used to train the ensemble.
    xvalid: np.ndarray
    yvalid: np.ndarray


def split_dataset(
    X, y, train_size=0.7, validation_size=0.15, test_size=0.15, random_state=None
):
    """
    Splits a dataset into training, validation, and test sets.

    Parameters
    ----------
    X : array-like
        Features dataset.
    y : array-like
        Labels dataset.
    train_size : float
        Proportion of the data to be used for training.
    validation_size : float
        Proportion of the data to be used for validation.
    test_size : float
        Proportion of the data to be used for testing.
    random_state: int, optional
        Random seed for reproducibility.

    Returns
    -------
        X_train: np.ndarray
        X_val: np.ndarray
        X_test: np.ndarray
        y_train: np.ndarray
        y_val: np.ndarray
        y_test: np.ndarray

    Examples
    --------
    >>> X = np.random.rand(100, 5)  # 100 samples, 5 features
    >>> y = np.random.randint(0, 2, 100)  # Binary target variable
    >>> X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    ....    X, y, train_size=0.7, validation_size=0.15, test_size=0.15, random_state=42)
    """
    # Ensure proportions sum to 1
    total = train_size + validation_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("train_size, validation_size, and test_size must sum to 1.")

    # Split into train and temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_size), random_state=random_state
    )

    # Calculate validation proportion of temp split
    val_proportion = validation_size / (validation_size + test_size)

    # Split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_proportion), random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def neg_root_mean_squared_error(ytrue, ypred):
    return -root_mean_squared_error(ytrue, ypred)


def count_nnz_leafs(at):
    nnz = 0
    for t in at:
        for lid in t.get_leaf_ids():
            lvals = t.get_leaf_values(lid)
            nnz += np.sum(np.abs(lvals) > 1e-5)
    return nnz


def pareto_front(items, m1, m2):
    """ metrics higher better """


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
        f" nnz {r.nnzleafs:5d}",
    )


def print_fit(r, alpha_search):
    import colorama

    RST = colorama.Style.RESET_ALL

    mtrain = alpha_search.mtrain_ref
    mvalid = alpha_search.mvalid_ref

    ctr = _color(r.clf_mtrain, mtrain, alpha_search)
    cte = _color(r.clf_mvalid, mvalid, alpha_search)

    status = f"{colorama.Fore.GREEN}fit ok{RST}"
    if alpha_search.underfits(r.clf_mtrain, r.clf_mvalid):
        status = f"{colorama.Fore.YELLOW}under {RST}"
    elif alpha_search.overfits(r.clf_mtrain, r.clf_mvalid):
        status = f"{colorama.Fore.RED}over  {RST}"

    print(f"{mtrain:7.3f} {mvalid:7.3f} ->", end=" ")

    ndigits = int(np.ceil(np.log10(1 + r.num_params)))

    print(
        f"{ctr}{r.clf_mtrain:7.3f}{RST} {cte}{r.clf_mvalid:7.3f}{RST},",
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
