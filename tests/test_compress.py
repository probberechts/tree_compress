import os
import numpy as np
import unittest
import veritas
import tree_compress

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BPATH = os.path.dirname(__file__)


class TestCompress(unittest.TestCase):
    def test_simple(self):
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

        clf = RandomForestClassifier(
            n_estimators=20,
            max_depth=6,
        )
        clf.fit(xtrain, ytrain)

        yhat1 = clf.predict(X)


        at = veritas.get_addtree(clf)
        yhat2 = at.predict(X) > 0.5

        data = tree_compress.Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)
        compressor = tree_compress.Compress(data, at)
        relerr = 0.02
        at_pruned = compressor.compress(relerr, max_rounds=1)
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
            plt.show()
        except ModuleNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
