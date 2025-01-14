import os

import numpy as np
import pytest
import veritas
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, root_mean_squared_error

from tree_compress import Compress, Data, LassoCompress
from tree_compress.util import split_dataset

BPATH = os.path.dirname(__file__)


@pytest.fixture
def dataset():
    """Fixture to prepare and provide the dataset."""
    img = np.load(os.path.join(BPATH, "data/img.npy"))
    X = np.array([[x, y] for x in range(100) for y in range(100)])
    y = np.array([img[x, y] for x, y in X])
    return X, y


@pytest.mark.parametrize(
    "clazz,kwargs",
    [
        (RandomForestClassifier, {}),
        (GradientBoostingClassifier, {"learning_rate": 0.8}),
    ],
)
def test_binary_classification(clazz, kwargs, dataset):
    """Test binary classification compression."""

    X, yreal = dataset

    # Convert labels to binary
    X = X.astype(veritas.FloatT)
    ymedian = np.median(yreal)
    yclf = yreal > ymedian

    # Split the dataset
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = split_dataset(
        X, yclf, train_size=0.6, validation_size=0.2, test_size=0.2, random_state=42
    )

    # Train the classifier
    clf = clazz(n_estimators=100, max_depth=6, **kwargs)
    clf.fit(xtrain, ytrain)

    # Predictions
    yhat_clf = clf.predict(X)

    # Convert to AddTree
    at = veritas.get_addtree(clf)
    veritas.test_conversion(at, xtrain[:10, :], clf.predict_proba(xtrain[:10, :])[:, 1])
    yhat_at = at.predict(X) > 0.5

    # Compress
    relerr = 0.01
    data = Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)
    compressor = Compress(
        data,
        at,
        metric=accuracy_score,
        isworse=lambda v, ref: v > ref + relerr,
    )
    at_pruned = compressor.compress(max_rounds=1)
    yhat_at_pruned = at_pruned.predict(X) > 0.5

    before = compressor.records[0]
    after = compressor.records[-1]

    assert before.ntrees > after.ntrees
    assert before.nnodes > after.nnodes
    assert before.nleafs > after.nleafs
    assert before.nnz_leafs > after.nnz_leafs
    assert after.mvalid + relerr >= before.mvalid

    # Optional: Visualization
    try:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(yclf.reshape((100, 100)))
        ax[0, 0].set_title("ground truth")
        ax[0, 1].imshow(yhat_clf.reshape((100, 100)))
        ax[0, 1].set_title("clf pred")
        ax[1, 0].imshow(yhat_at.reshape((100, 100)))
        ax[1, 0].set_title("at")
        ax[1, 1].imshow(yhat_at_pruned.reshape((100, 100)))
        ax[1, 1].set_title("pruned")
        plt.tight_layout()
        plt.savefig(f"test_compress_binary_classification_{clazz.__name__}.png")
        plt.show()
    except ImportError:
        pass


@pytest.mark.parametrize(
    "clazz,kwargs",
    [
        (RandomForestClassifier, {}),
        (GradientBoostingClassifier, {"learning_rate": 0.8}),
    ],
)
def test_multiclass_classification(clazz, kwargs, dataset):
    """Test multiclass classification compression."""
    X, yreal = dataset

    # Create multiclass targets using quantiles
    quantiles = np.quantile(yreal, [0.33, 0.67])
    y = np.digitize(yreal, quantiles).astype(int)
    X = X.astype(veritas.FloatT)

    # Split the dataset
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = split_dataset(
        X, y, train_size=0.6, validation_size=0.2, test_size=0.2, random_state=42
    )

    # Train the classifier
    clf = clazz(random_state=37, n_estimators=20, max_depth=5, **kwargs)
    clf.fit(xtrain, ytrain)

    # Predictions
    yhat_clf = clf.predict(X)

    # Convert to AddTree
    at = veritas.get_addtree(clf)
    veritas.test_conversion(at, X[:10, :], clf.predict_proba(X[:10, :]))
    yhat_at = np.argmax(at.predict(X), axis=1)

    # Compression
    abserr = 0.01
    data = Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)
    compressor = Compress(
        data,
        at,
        metric=accuracy_score,
        isworse=lambda v, ref: v - ref > abserr,
        silent=False,
    )
    compressor.no_convergence_warning = True
    at_pruned = compressor.compress(max_rounds=2)
    yhat_at_pruned = np.argmax(at_pruned.predict(X), axis=1)

    # Validate compression statistics
    before = compressor.records[0]
    after = compressor.records[-1]

    assert before.ntrees >= after.ntrees
    assert before.nnodes > after.nnodes
    assert before.nleafs > after.nleafs
    assert before.nnz_leafs > after.nnz_leafs
    assert after.mvalid + abserr >= before.mvalid

    # Optional visualization
    try:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(y.reshape((100, 100)), interpolation="none")
        ax[0, 0].set_title("ground truth")
        ax[0, 1].imshow(yhat_clf.reshape((100, 100)), interpolation="none")
        ax[0, 1].set_title("clf pred")
        ax[1, 0].imshow(yhat_at.reshape((100, 100)), interpolation="none")
        ax[1, 0].set_title("at (check)")
        ax[1, 1].imshow(yhat_at_pruned.reshape((100, 100)), interpolation="none")
        ax[1, 1].set_title("compressed")
        ax[0, 2].imshow(
            (yhat_at == y).reshape((100, 100)), cmap="Reds", interpolation="none"
        )
        ax[0, 2].set_title("errors clf")
        ax[1, 2].imshow(
            (yhat_at_pruned == y).reshape((100, 100)), cmap="Reds", interpolation="none"
        )
        ax[1, 2].set_title("errors compressed")

        fig, ax = plt.subplots(3, 3)
        for k in range(3):
            ax[k, 0].set_title(f"ground target {k}")
            ax[k, 0].imshow((y == k).reshape((100, 100)), interpolation="none")
            ax[k, 1].set_title(f"uncompr. target {k}")
            ax[k, 1].imshow(
                at.eval(X)[:, k].reshape((100, 100)),
                vmin=-1,
                vmax=1,
                interpolation="none",
            )
            ax[k, 2].set_title(f"compr. target {k}")
            ax[k, 2].imshow(
                at_pruned.eval(X)[:, k].reshape((100, 100)),
                vmin=-1,
                vmax=1,
                interpolation="none",
            )
        plt.tight_layout()
        plt.savefig(f"test_compress_multiclass_classification_{clazz.__name__}.png")
        plt.show()
    except ImportError:
        pass


@pytest.mark.parametrize(
    "clazz,kwargs",
    [
        (RandomForestRegressor, {}),
        (GradientBoostingRegressor, {"learning_rate": 0.8}),
    ],
)
def test_regression(clazz, kwargs, dataset):
    """Test regression compression."""
    X, y = dataset

    y = y.astype(veritas.FloatT) / float(np.max(y))
    X = X.astype(veritas.FloatT)

    # Split the dataset
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = split_dataset(
        X, y, train_size=0.6, validation_size=0.2, test_size=0.2, random_state=42
    )

    # Train the regressor
    clf = clazz(n_estimators=40, max_depth=4, **kwargs)
    clf.fit(xtrain, ytrain)
    yhat_clf = clf.predict(X)

    # Convert to AddTree
    at_orig = veritas.get_addtree(clf)
    veritas.test_conversion(at_orig, xtrain[:10, :], clf.predict(xtrain[:10, :]))
    yhat_at = at_orig.predict(X)

    # Compression
    abserr = np.mean(np.abs(ytrain - yhat_at)) / 20.0
    data = Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)
    compr = LassoCompress(
        data,
        at_orig,
        metric=root_mean_squared_error,
        isworse=lambda v, ref: v - ref > abserr,
        linclf_type="Lasso",
        seed=123,
        silent=False,
    )
    at_compr = compr.compress(max_rounds=1)
    yhat_at_pruned = at_compr.predict(X)

    before = compr.records[0]
    after = compr.records[-1]

    assert before.ntrees >= after.ntrees
    assert before.nnodes > after.nnodes
    assert before.nleafs > after.nleafs
    assert before.nnz_leafs > after.nnz_leafs

    # Optional visualization
    try:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(y.reshape((100, 100)))
        ax[0, 0].set_title("ground truth")
        ax[0, 1].imshow(yhat_clf.reshape((100, 100)))
        ax[0, 1].set_title("clf pred")
        ax[1, 0].imshow(yhat_at.reshape((100, 100)))
        ax[1, 0].set_title("at")
        ax[1, 1].imshow(yhat_at_pruned.reshape((100, 100)))
        ax[1, 1].set_title("compressed")
        plt.tight_layout()
        plt.savefig(f"test_compress_regression_{clazz.__name__}.png")
        plt.show()
    except ImportError:
        pass
