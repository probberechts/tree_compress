import os
import numpy as np
import unittest
import tree_compress
import veritas

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

BPATH = os.path.dirname(__file__)


class TestCompress(unittest.TestCase):
    def test_binary_rf(self):
        self.do_binary(RandomForestClassifier)

    def test_binary_gbdt(self):
        self.do_binary(GradientBoostingClassifier, learning_rate=0.8)

    def do_binary(self, clazz, **kwargs):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        yreal = np.array([img[x, y] for x, y in X])
        X = X.astype(veritas.FloatT)

        ymedian = np.median(yreal)
        yclf = yreal > ymedian

        xtrain, xtest, ytrain, ytest = train_test_split(
            X, yclf, test_size=0.40, random_state=42
        )
        xvalid, xtest, yvalid, ytest = train_test_split(xtest, ytest, test_size=0.5)

        clf = clazz(
            n_estimators=100,
            max_depth=6,
            **kwargs
        )
        clf.fit(xtrain, ytrain)

        yhat1 = clf.predict(X)

        at = veritas.get_addtree(clf)
        veritas.test_conversion(at, X[:10,:], clf.predict_proba(X[:10,:])[:,1])
        yhat2 = at.predict(X) > 0.5

        data = tree_compress.Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)
        compressor = tree_compress.Compress(data, at)
        relerr = 0.01
        at_pruned = compressor.compress(relerr=relerr, max_rounds=1)
        yhat3 = at_pruned.predict(X) > 0.5

        before = compressor.records[0]
        after = compressor.records[-1]

        self.assertGreater(before.ntrees, after.ntrees)
        self.assertGreater(before.nnodes, after.nnodes)
        self.assertGreater(before.nleafs, after.nleafs)
        self.assertGreater(before.nnz_leafs, after.nnz_leafs)
        self.assertGreaterEqual(after.mvalid+relerr, before.mvalid)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(yclf.reshape((100, 100)))
            ax[0, 0].set_title("ground truth")
            ax[0, 1].imshow(yhat1.reshape((100, 100)))
            ax[0, 1].set_title("clf pred")
            ax[1, 0].imshow(yhat2.reshape((100, 100)))
            ax[1, 0].set_title("at")
            ax[1, 1].imshow(yhat3.reshape((100, 100)))
            ax[1, 1].set_title("pruned")
            plt.tight_layout()
            plt.show()
        except ModuleNotFoundError:
            pass

    def test_regression_rf(self):
        self.do_regression(RandomForestRegressor)

    def test_regression_gbdt(self):
        self.do_regression(GradientBoostingRegressor, learning_rate=0.8)

    def do_regression(self, clazz, **kwargs):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        y = np.array([img[x, y] for x, y in X])
        y = y.astype(veritas.FloatT) / float(np.max(y))
        X = X.astype(veritas.FloatT)

        xtrain, xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.40, random_state=42
        )
        xvalid, xtest, yvalid, ytest = train_test_split(xtest, ytest, test_size=0.5)

        clf = clazz(
            n_estimators=40,
            max_depth=6,
            **kwargs
        )
        clf.fit(xtrain, ytrain)

        yhat1 = clf.predict(X)


        at = veritas.get_addtree(clf)
        veritas.test_conversion(at, X[:10,:], clf.predict(X[:10,:]))
        yhat2 = at.predict(X)

        data = tree_compress.Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)
        compressor = tree_compress.Compress(data, at, silent=False)
        compressor.no_convergence_warning = True
        relerr = 0.2
        at_pruned = compressor.compress(relerr=relerr, max_rounds=1)
        yhat3 = at_pruned.predict(X)

        before = compressor.records[0]
        after = compressor.records[-1]

        self.assertGreaterEqual(before.ntrees, after.ntrees)
        self.assertGreater(before.nnodes, after.nnodes)
        self.assertGreater(before.nleafs, after.nleafs)
        self.assertGreater(before.nnz_leafs, after.nnz_leafs)
        self.assertGreaterEqual(-(1+relerr)*after.mvalid, -before.mvalid)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(y.reshape((100, 100)))
            ax[0, 0].set_title("ground truth")
            ax[0, 1].imshow(yhat1.reshape((100, 100)))
            ax[0, 1].set_title("clf pred")
            ax[1, 0].imshow(yhat2.reshape((100, 100)))
            ax[1, 0].set_title("at")
            ax[1, 1].imshow(yhat3.reshape((100, 100)))
            ax[1, 1].set_title("pruned")
            plt.tight_layout()
            plt.show()
        except ModuleNotFoundError:
            pass

    def test_multiclass_rf(self):
        self.do_multiclass(RandomForestClassifier)

    def test_multiclass_gbdt(self):
        self.do_multiclass(GradientBoostingClassifier)

    def do_multiclass(self, clazz, **kwargs):
        img = np.load(os.path.join(BPATH, "data/img.npy"))
        X = np.array([[x, y] for x in range(100) for y in range(100)])
        yreal = np.array([img[x, y] for x, y in X])

        quantiles = np.quantile(yreal, [0.33, 0.67])
        y = np.digitize(yreal, quantiles)
        y = y.astype(int)
        X = X.astype(veritas.FloatT)

        xtrain, xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.40, random_state=42
        )
        xvalid, xtest, yvalid, ytest = train_test_split(xtest, ytest, test_size=0.5)

        clf = clazz(
            n_estimators=20,
            max_depth=4,
            **kwargs
        )
        clf.fit(xtrain, ytrain)

        yhat1 = clf.predict(X)


        at = veritas.get_addtree(clf)
        veritas.test_conversion(at, X[:10,:], clf.predict_proba(X[:10,:]))
        yhat2 = np.argmax(at.predict(X), axis=1)

        data = tree_compress.Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)
        compressor = tree_compress.Compress(data, at, silent=False)
        relerr = 0.02
        at_pruned = compressor.compress(relerr=relerr, max_rounds=1)
        yhat3 = np.argmax(at_pruned.predict(X), axis=1)
        before = compressor.records[0]
        after = compressor.records[-1]

        self.assertGreaterEqual(before.ntrees, after.ntrees)
        self.assertGreater(before.nnodes, after.nnodes)
        self.assertGreater(before.nleafs, after.nleafs)
        self.assertGreater(before.nnz_leafs, after.nnz_leafs)
        self.assertGreaterEqual((1+relerr)*after.mvalid, before.mvalid)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 2)
            ax[0, 0].imshow(y.reshape((100, 100)))
            ax[0, 0].set_title("ground truth")
            ax[0, 1].imshow(yhat1.reshape((100, 100)))
            ax[0, 1].set_title("clf pred")
            ax[1, 0].imshow(yhat2.reshape((100, 100)))
            ax[1, 0].set_title("at")
            ax[1, 1].imshow(yhat3.reshape((100, 100)))
            #ax[1, 1].imshow(at.predict(X)[:,2].reshape((100, 100)))
            ax[1, 1].set_title("pruned")
            plt.tight_layout()
            plt.show()
        except ModuleNotFoundError:
            pass

if __name__ == "__main__":
    unittest.main()
