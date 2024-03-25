# TreeCompress: top-down compression of decision tree ensembles using L1 regularization

This Python software package takes a pre-trained tree ensemble model and compresses
it in a top-down fashion. Starting from the root and going down level per level, it prunes
away subtrees by fitting coefficients.

It uses the tree representation of [Veritas](https://github.com/laudv/veritas).

## Installation

From source:
```bash
git clone https://github.com/laudv/tree_compress.git
cd tree_compress
pip install .
```

## Example

```python
import tree_compress
import veritas
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

noise = 0.05
xtrain, ytrain = make_moons(200, noise=noise, random_state=1)
xtest, ytest = make_moons(200, noise=noise, random_state=2)
xvalid, yvalid = make_moons(200, noise=noise, random_state=3)

data = tree_compress.Data(xtrain, ytrain, xtest, ytest, xvalid, yvalid)

clf = RandomForestClassifier(
        max_depth=5,
        random_state=2,
        n_estimators=50)
clf.fit(data.xtrain, data.ytrain)
at_pruned = tree_compress.compress_topdown(data, at_orig, relerr=0.02, max_rounds=2)
```

See full example [here](./examples/two_moons.ipynb).
