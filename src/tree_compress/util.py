from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from veritas import AddTree, AddTreeType


@dataclass
class Data:
    """
    Container for training, testing, and validation data and their respective labels.

    Attributes
    ----------
    xtrain : np.ndarray
        The data used to train the tree ensemble.
    ytrain : np.ndarray
        Labels corresponding to `xtrain`.

    xtest : np.ndarray
        The data used to evaluate the tree ensemble and the pruned tree ensemble.
        This data is **not** used during the learning or compression process.
    ytest : np.ndarray
        Labels corresponding to `xtest`.

    xvalid : np.ndarray
        The data used to tune the strength of the regularization coefficient, alpha.
        This data should not have been used to train the ensemble, to avoid
        overfitting on the training data.
    yvalid : np.ndarray
        Labels corresponding to `xvalid`.
    """

    xtrain: np.ndarray
    ytrain: np.ndarray

    xtest: np.ndarray
    ytest: np.ndarray

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


def count_nnz_leafs(at: AddTree) -> int:
    """Count the number of non-zero leaf values in a veritas.AddTree."""
    nnz = 0
    for t in at:
        for lid in t.get_leaf_ids():
            lvals = t.get_leaf_values(lid)
            nnz += np.sum(np.abs(lvals) > 1e-5)
    return nnz


def at_isregr(at: AddTree) -> bool:
    """Check if a given veritas.AddTree is a regressor."""
    return at.get_type() in {
        AddTreeType.REGR,
        AddTreeType.REGR_MEAN,
    }


def at_isclf(at: AddTree) -> bool:
    """Check if a given veritas.AddTree is a classifier."""
    return at.get_type() in {
        AddTreeType.CLF_SOFTMAX,
        AddTreeType.CLF_MEAN,
    }


def at_predlab(at: AddTree, x: np.ndarray) -> np.ndarray:
    """Predict hard labels for veritas.AddTree"""
    if at_isregr(at):
        return at.predict(x)
    elif at_isclf(at):
        if at.num_leaf_values() == 1:
            return (at.eval(x)[:, 0] >= 0.0).astype(int)
        else:
            return at.eval(x).argmax(axis=1)
    else:
        raise RuntimeError("cannot determine task")


def default_metric(
    at: AddTree,
) -> Tuple[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Return default metric and metric function for given AddTree."""
    if at_isregr(at):
        return "neg_rmse", lambda ytrue, ypred: -root_mean_squared_error(ytrue, ypred)
    elif at_isclf(at):
        return "accuracy", accuracy_score
    else:
        raise RuntimeError("cannot determine task")


def isworse_abserr(
    metric: float,
    reference: float,
    abserr: float = 0.0,
    higher_is_better: bool = True,
) -> bool:
    """
    Check if a metric is worse than a reference by an absolute error threshold.

    Parameters
    ----------
    metric : float
        The current metric value.
    reference : float
        The reference metric value to compare against.
    abserr : float, optional
        The absolute error threshold. Defaults to 0.0.
    higher_is_better : bool, optional
        If True, higher metric values are considered better.
        If False, lower metric values are considered better. Defaults to True.

    Returns
    -------
    bool
        True if the metric is worse than the reference by the given absolute
        error threshold, False otherwise.

    Examples
    --------
    >>> isworse_abserr(90, 100, abserr=5, higher_is_better=True)
    True
    >>> isworse_abserr(95, 100, abserr=5, higher_is_better=True)
    False
    >>> isworse_abserr(105, 100, abserr=5, higher_is_better=False)
    True
    >>> isworse_abserr(95, 100, abserr=5, higher_is_better=False)
    False
    """
    diff = metric - reference
    if higher_is_better:
        return diff <= -abserr
    else:
        return diff >= abserr


def isworse_relerr(
    metric: float,
    reference: float,
    relerr: float = 0.0,
    higher_is_better: bool = True,
) -> bool:
    """
    Check if a metric is worse than a reference by a relative error threshold.

    Parameters
    ----------
    metric : float
        The current metric value.
    reference : float
        The reference metric value to compare against.
    relerr : float, optional
        The relative error threshold. Defaults to 0.0.
    higher_is_better : bool, optional
        If True, higher metric values are considered better.
        If False, lower metric values are considered better. Defaults to True.

    Returns
    -------
    bool
        True if the metric is worse than the reference by the given relative
        error threshold, False otherwise.

    Examples
    --------
    >>> isworse_relerr(90, 100, relerr=0.1, higher_is_better=True)
    True
    >>> isworse_relerr(110, 100, relerr=0.1, higher_is_better=True)
    False
    >>> isworse_relerr(110, 100, relerr=0.1, higher_is_better=False)
    True
    >>> isworse_relerr(90, 100, relerr=0.1, higher_is_better=False)
    False
    """
    eps = (metric - reference) / abs(reference)
    if higher_is_better:
        return eps <= -relerr
    else:
        return eps >= relerr


def is_almost_eq(metric: float, reference: float, relerr: float = 1e-5) -> bool:
    """
    Check if two values are approximately equal within a relative error threshold.

    Parameters
    ----------
    metric : float
        The current metric value.
    reference : float
        The reference value to compare against.
    relerr : float, optional
        The relative error threshold for comparison. Defaults to 1e-5.

    Returns
    -------
    bool
        True if the relative error between `metric` and `reference` is less than
        `relerr`, False otherwise.

    Examples
    --------
    >>> is_almost_eq(1.00001, 1.0, relerr=1e-5)
    True
    >>> is_almost_eq(1.001, 1.0, relerr=1e-5)
    False
    >>> is_almost_eq(0.99999, 1.0, relerr=1e-5)
    True
    """
    eps = abs((metric - reference) / reference)
    return eps < relerr


def is_not_almost_eq(metric: float, reference: float, relerr: float = 1e-5) -> bool:
    """
    Check if two values are not approximately equal within a relative error threshold.

    Parameters
    ----------
    metric : float
        The current metric value.
    reference : float
        The reference value to compare against.
    relerr : float, optional
        The relative error threshold for comparison. Defaults to 1e-5.

    Returns
    -------
    bool
        True if the relative error between `metric` and `reference` is greater
        than or equal to `relerr`, False otherwise.

    Examples
    --------
    >>> is_not_almost_eq(1.00001, 1.0, relerr=1e-5)
    False
    >>> is_not_almost_eq(1.001, 1.0, relerr=1e-5)
    True
    >>> is_not_almost_eq(0.99999, 1.0, relerr=1e-5)
    False
    """
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
