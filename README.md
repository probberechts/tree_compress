# TreeCompress: top-down compression of decision tree ensembles using L1 regularization

## Installation

From _PyPI_:

```bash
# activate your environment
source my_virtual_environment/bin/activate

pip install dtai-tree-compress
```

From source:
```bash
git clone https://github.com/ dtai-tree-compress
cd dtai-tree-compress

# activate your environment
source my_virtual_environment/bin/activate

pip install .
```

## Example

```python
import tree_compress
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier

(X,Y) = make_moons(100)

clf = RandomForestClassifier(
        max_depth=4,
        random_state=0,
        n_estimators=50)

clf.fit(X, Y)
at = veritas.get_addtree(clf)

at_pruned = tree_compress.compress(at, relerr=0.02)
```
