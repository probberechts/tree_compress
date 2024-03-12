import time
import numpy as np
import veritas
import itertools

from dataclasses import dataclass
from .util import count_nnz_leafs, metric, print_metrics, print_fit, isworse
from scipy.sparse import csr_matrix, csc_matrix
from functools import partial
from sklearn.linear_model import LogisticRegression, Lasso


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


@dataclass(init=False)
class CompressRecord:
    level: int
    alpha: float
    tindex: float
    ttransform: float
    tsearch: float

    at: veritas.AddTree

    ntrees: int
    nnodes: int
    nleafs: int
    nnz_leafs: int

    mtrain: float
    mtest: float
    mvalid: float
    clf_mtrain: float
    clf_mvalid: float

    def __init__(self, level, alpha, at):
        self.level = level
        self.alpha = alpha
        self.tindex = 0.0
        self.ttransform = 0.0
        self.tsearch = 0.0

        self.at = at

        self.ntrees = len(at)
        self.nnodes = at.num_nodes()
        self.nleafs = at.num_leafs()
        self.nnz_leafs = count_nnz_leafs(at)

        self.mtrain = 0.0
        self.mtest = 0.0
        self.mvalid = 0.0

        self.clf_mtrain = 0.0
        self.clf_mvalid = 0.0


@dataclass(init=False)
class AlphaRecord:
    lo: float
    hi: float
    alpha: float

    mtrain_clf: float
    mvalid_clf: float
    num_params: int
    num_removed: int
    num_kept: int
    frac_removed: float
    fit_time: float

    intercept: np.ndarray
    coefs: np.ndarray

    def __init__(self, lo, hi, alpha):
        self.lo = lo
        self.hi = hi
        self.alpha = alpha
        self.mtrain_clf = 0.0
        self.mvalid_clf = 0.0
        self.num_params = 0
        self.num_removed = 0
        self.num_kept = 0
        self.frac_removed = 0.0
        self.fit_time = 0.0
        self.intercept = None
        self.coefs = None


class AlphaSearch:
    def __init__(self, compress, relerr):
        self.compress = compress
        self.relerr = relerr

        self.round = 0
        self.step = 1
        self.lo = -3
        self.hi = 4

        self.records = []

        self.set_lohis()

    def __iter__(self):
        return self

    def __next__(self):
        nsteps = self.nsteps()
        if self.step > nsteps:  # next round
            lo, hi = self.next_lohis()

            self.lo, self.hi = lo, hi
            self.set_lohis()
            self.step = 1

        lo, mid, hi = self.lohis[self.step - 1 : self.step + 2]
        alpha = np.power(10.0, mid)

        record = AlphaRecord(lo, hi, alpha)
        self.records.append(record)

        self.step += 1
        return record

    def isworse_tr(self, mtrain):
        return isworse(mtrain, self.compress.mtrain, relerr=self.relerr)

    def isworse_va(self, mvalid):
        return isworse(mvalid, self.compress.mvalid, relerr=self.relerr)

    def isnotworse_tr(self, mtrain):
        return not isworse(mtrain, self.compress.mtrain, relerr=self.relerr)

    def isnotworse_va(self, mvalid):
        return not isworse(mvalid, self.compress.mvalid, relerr=self.relerr)

    def overfits(self, mtrain, mvalid):
        cond1 = self.isnotworse_tr(mtrain)
        cond2 = self.isworse_va(mvalid)
        cond3 = isworse(mvalid, mtrain, self.relerr)
        return cond1 and cond2 and cond3

    def underfits(self, mtrain, mvalid):
        cond1 = self.isworse_tr(mtrain)
        cond2 = self.isworse_va(mvalid)
        return cond1 and cond2

    def nsteps(self):
        return self.compress.alpha_search_round_steps[self.round]

    def set_lohis(self):
        nsteps = self.nsteps()
        self.lohis = np.linspace(self.lo, self.hi, nsteps + 2)

    def quality_filter(self, records):
        filt = filter(
            lambda r: self.isnotworse_va(r.mvalid_clf)
            and r.frac_removed < 1.0
            and not isworse(r.mtrain_clf, r.mvalid_clf),  # overfitting
            records,
        )
        return filt

    def next_lohis(self):
        nsteps = self.nsteps()
        num_rounds = len(self.compress.alpha_search_round_steps)

        self.round += 1
        if self.round >= num_rounds:
            raise StopIteration()

        prev_round_records = self.records[-1 * nsteps :]

        # Out of the records whose validation metric is good enough wrt self.relerr...
        filt = self.quality_filter(prev_round_records)
        # ... pick the one with the highest alpha
        best = max(filt, default=None, key=lambda r: r.alpha)

        # If nothing was good enough, find the last record that was overfitting and the
        # first that was underfitting, and look at that transition in more detail
        r_over, r_under = None, None
        if best is None:
            for r in prev_round_records:
                if self.overfits(r.mtrain_clf, r.mvalid_clf):
                    r_over = r
                elif r_over is not None and self.underfits(r.mtrain_clf, r.mvalid_clf):
                    r_under = r
                    break
            if r_over is not None and r_under is not None:
                lo = (r_over.lo + r_over.hi) / 2.0
                hi = (r_under.lo + r_under.hi) / 2.0
                return lo, hi
            else:
                raise StopIteration()
        else:
            return best.lo, best.hi

    def get_best_record(self):
        # Out of the good enough solutions (wrt self.relerr) ...
        filt = self.quality_filter(self.records)
        return max(filt, default=None, key=lambda r: r.frac_removed)


class Compress:
    def __init__(self, data, at, silent=False, seed=569):
        self.d = data
        self.at = at
        self.silent = silent
        self.seed = seed

        self.mtrain = metric(self.at, ytrue=self.d.ytrain, x=self.d.xtrain)
        self.mtest = metric(self.at, ytrue=self.d.ytest, x=self.d.xtest)
        self.mvalid = metric(self.at, ytrue=self.d.yvalid, x=self.d.xvalid)

        self.alpha_search_round_steps = [16, 8, 4]

        if not self.silent:
            print(
                "MODEL PERF:",
                f"mtr {self.mtrain:.3f} mte {self.mtest:.3f} mva {self.mvalid:.3f}",
            )

        start_record = CompressRecord(-1, 0.0, self.at)
        start_record.mtrain = self.mtrain
        start_record.mtest = self.mtest
        start_record.mvalid = self.mvalid
        self.records = [start_record]

    def is_regression(self):
        return self.at.get_type() in {
            veritas.AddTreeType.REGR,
            veritas.AddTreeType.REGR_MEAN,
        }

    def _get_index(self, level, include_higher_leaves):
        index = []  # tree_index -> node_idx -> col_idex
        num_cols = 0

        for t in self.at:
            index0 = {}  # leaf_id -> node_id at level
            index1 = {}  # node_id at level -> xxmatrix column index

            # make an index: leaf_id -> [nodes along root-to-leaf path]
            for leaf_id in t.get_leaf_ids():
                n = leaf_id
                path = [n]
                while not t.is_root(n):
                    n = t.parent(n)
                    path.insert(0, n)

                # if this leaf is higher up than the level, either just take the leaf,
                if include_higher_leaves:
                    n_at_level = path[level] if len(path) > level else path[-1]

                # ... or do not include it this time.
                else:
                    n_at_level = path[level] if len(path) > level else None

                if n_at_level is not None:
                    index0[leaf_id] = n_at_level

                    if n_at_level not in index1:
                        index1[n_at_level] = num_cols
                        if t.is_root(n_at_level):
                            num_cols += 1
                        elif t.is_leaf(n_at_level):
                            num_cols += 1
                        else:
                            num_cols += 2

            index.append((index0, index1))
        return index, num_cols

    def _transform_data(self, x, index, num_cols):
        row_ind, col_ind = [], []
        values = []
        num_rows = x.shape[0]

        for t, (index0, index1) in zip(self.at, index):
            for i, leaf_id in enumerate(t.eval_node(x)):
                if leaf_id not in index0:
                    continue

                n_at_level = index0[leaf_id]
                if t.is_root(n_at_level) or t.is_leaf(n_at_level):
                    row_ind += [i]
                    col_idx = index1[n_at_level]
                    col_ind += [col_idx]
                    if t.is_root(n_at_level):
                        # a coefficient * leaf values for root nodes, no bias
                        values += [t.get_leaf_value(leaf_id, 0)]
                    else:
                        values += [1.0]  # only bias term for leaves
                else:
                    row_ind += [i, i]
                    col_idx = index1[n_at_level]
                    col_ind += [col_idx, col_idx + 1]
                    values += [1.0, t.get_leaf_value(leaf_id, 0)]

        if self.is_regression():  # Lasso
            xx = csc_matrix(
                (np.array(values), (row_ind, col_ind)), shape=(num_rows, num_cols)
            )
        else:
            xx = csr_matrix(
                (np.array(values), (row_ind, col_ind)), shape=(num_rows, num_cols)
            )

        frac_nnz = xx.nnz / np.prod(xx.shape) if num_cols > 0 else 1.0

        if frac_nnz > 0.01:
            xx = xx.toarray()
        return xx

    def compress(self, relerr, max_rounds=4):
        last_record = self.records[-1]
        for i in range(max_rounds):
            if not self.silent:
                print(f"\n\nROUND {i+1}")

            self.compress_round(relerr)

            new_record = self.records[-1]
            has_improved = last_record.nnodes <= new_record.nnodes
            last_record = new_record

            if has_improved:
                break
        return last_record.at

    def compress_round(self, relerr):
        isworse_cmp = partial(isworse, relerr=relerr)

        for level in itertools.count():
            r = self.compress_level(level, relerr)

            if not self.silent:
                r0 = self.records[0]
                r1 = self.records[-1]
                print_metrics("orig", r0)
                print_metrics("prev", r1, rcmp=r0, cmp=isworse_cmp)
                print_metrics("now", r, rcmp=r0, cmp=isworse_cmp)
                print()

            self.records.append(r)

            max_depth = max(
                max(tree.depth(i) for i in tree.get_leaf_ids()) for tree in self.at
            )
            if level >= max_depth:
                if not self.silent:
                    print(f"DONE, depth of tree reached {level}, {max_depth}")
                break

        return self.records[-1].at

    def compress_level(self, level, relerr):
        tindex = time.time()
        index, num_cols = self._get_index(level, True)
        tindex = time.time() - tindex

        ttransform = time.time()
        xxtrain = self._transform_data(self.d.xtrain, index, num_cols)
        xxvalid = self._transform_data(self.d.xvalid, index, num_cols)
        ttransform = time.time() - ttransform

        if not self.silent:
            print(
                f"Level {level} xxtrain.shape {xxtrain.shape}",
                "dense" if isinstance(xxtrain, np.ndarray) else "sparse",
                f"transform time: {tindex:.2f}s, {ttransform:.2f}s",
                f"relerr {relerr}",
            )
        alpha_search = AlphaSearch(self, relerr)
        clf = self._get_regularized_lin_clf(xxtrain)

        if num_cols == 0:
            raise RuntimeError("no columns? everything is pruned away?")

        tsearch = time.time()
        for alpha_record in alpha_search:
            self._update_lin_clf_alpha(clf, alpha_record.alpha)
            self.fit_coefficients(clf, xxtrain, xxvalid, alpha_record)

            if not self.silent:
                print_fit(alpha_record, alpha_search)

        tsearch = time.time() - tsearch

        best = alpha_search.get_best_record()
        best_alpha = 0.0
        clf_mtrain = self.records[-1].clf_mtrain
        clf_mvalid = self.records[-1].clf_mvalid

        if best is not None:
            intercept = best.intercept
            coefs = best.coefs
            self.at = self.prune_trees(intercept, coefs, index)
            clf_mtrain = best.mtrain_clf
            clf_mvalid = best.mvalid_clf

        # record
        mtrain_prun = metric(self.at, ytrue=self.d.ytrain, x=self.d.xtrain)
        mtest_prun = metric(self.at, ytrue=self.d.ytest, x=self.d.xtest)
        mvalid_prun = metric(self.at, ytrue=self.d.yvalid, x=self.d.xvalid)
        record = CompressRecord(level, best_alpha, self.at)
        record.mtrain = mtrain_prun
        record.mtest = mtest_prun
        record.mvalid = mvalid_prun
        record.clf_mtrain = clf_mtrain
        record.clf_mvalid = clf_mvalid
        record.tindex = tindex
        record.ttransform = ttransform
        record.tsearch = tsearch

        return record

    def fit_coefficients(self, clf, xxtrain, xxvalid, alpha_record):
        fit_time = time.time()
        clf.fit(xxtrain, self.d.ytrain)
        fit_time = time.time() - fit_time

        num_params = xxtrain.shape[1]
        num_removed = np.sum(np.abs(clf.coef_) < 1e-6)
        frac_removed = num_removed / num_params

        alpha_record.mtrain_clf = metric(
            self.at, ytrue=self.d.ytrain, ypred=clf.predict(xxtrain)
        )
        alpha_record.mvalid_clf = metric(
            self.at, ytrue=self.d.yvalid, ypred=clf.predict(xxvalid)
        )
        alpha_record.num_params = num_params
        alpha_record.num_removed = num_removed
        alpha_record.num_kept = num_params - num_removed
        alpha_record.frac_removed = frac_removed
        alpha_record.intercept = np.copy(clf.intercept_)
        alpha_record.coefs = np.copy(clf.coef_)
        alpha_record.fit_time = fit_time

    def _get_regularized_lin_clf(self, xxtrain):
        self.seed += 1

        if self.is_regression():
            precompute = isinstance(xxtrain, np.ndarray)  # dense only
            return Lasso(
                alpha=1.0,
                random_state=self.seed,
                max_iter=5_000,
                tol=1e-4,
                warm_start=True,
                precompute=precompute,
            )
        else:
            return LogisticRegression(
                penalty="l1",
                C=1.0,
                solver="liblinear",
                max_iter=5_000,
                tol=1e-4,
                n_jobs=1,
                random_state=self.seed,
                warm_start=True,
            )

    def _update_lin_clf_alpha(self, clf, alpha):
        if alpha <= 0.0:
            raise RuntimeError("alpha == 0.0?")

        if self.is_regression():
            assert isinstance(clf, Lasso)
            clf.alpha = 0.0001 * alpha
        else:
            assert clf.penalty == "l1"
            clf.C = 1.0 / alpha

    def prune_trees(self, intercept, coefs, index):
        # TODO multiclass

        if self.is_regression():
            base_score = intercept
            coefs = coefs
            at_type = veritas.AddTreeType.REGR
        else:
            base_score = intercept[0]
            coefs = coefs[0, :]
            at_type = veritas.AddTreeType.CLF_SOFTMAX
        M = len(self.at)

        atp = veritas.AddTree(self.at.num_leaf_values(), at_type)
        atpp = atp.copy()
        offset = 0

        for m, t, (index0, index1) in zip(range(M), self.at, index):
            # special case: if coef of full tree is 0.0, then drop the tree
            if t.root() in index1:
                offset = index1[t.root()]
                coef = coefs[offset]
                if coef == 0.0:
                    continue
            self._copy_tree(t, atp.add_tree(), coefs, index1)

            tp = atp[len(atp) - 1]
            pruner = _TreeZeroLeafPruner(tp)
            if not pruner.is_root_zero():
                tpp = atpp.add_tree()
                pruner.prune(tp.root(), tpp, tpp.root())

        atp.set_base_score(0, base_score)
        atpp.set_base_score(0, base_score)
        return atpp

    def _copy_tree(self, t, tc, coefs, index1):
        self._copy_subtree(t, t.root(), tc, tc.root(), coefs, index1)

    def _copy_subtree(self, t, n, tc, nc, coefs, index1):
        stack = [(n, nc, 1.0, 0.0)]
        while len(stack) > 0:
            n, nc, coef, bias = stack.pop()

            if n in index1:
                assert coef == 1.0
                assert bias == 0.0

                offset = index1[n]
                if t.is_root(n):
                    bias, coef = 0.0, coefs[offset]
                elif t.is_leaf(n):
                    bias, coef = coefs[offset], 0.0
                else:
                    bias, coef = coefs[offset : offset + 2]

                if coef == 0.0:  # skip the branch, just predict bias
                    # print(f"cutting off branch {n}, leaf value {bias:.3f}")
                    tc.set_leaf_value(nc, 0, bias)
                    continue

            if t.is_internal(n):
                s = t.get_split(n)
                tc.split(nc, s.feat_id, s.split_value)
                stack.append((t.right(n), tc.right(nc), coef, bias))
                stack.append((t.left(n), tc.left(nc), coef, bias))
            else:
                leaf_value = bias + coef * t.get_leaf_value(n, 0)
                tc.set_leaf_value(nc, 0, leaf_value)


class _TreeZeroLeafPruner:
    def __init__(self, t):
        self.t = t
        self.is_zero = np.zeros(t.num_nodes(), dtype=bool)
        self._can_prune(t.root())

    # Mark which subtrees all have leaf values equal, and can be pruned
    def _can_prune(self, n):
        t = self.t
        if t.is_leaf(n):
            is_zero = t.get_leaf_value(n, 0) == 0.0
        else:
            is_zero_right = self._can_prune(t.right(n))
            is_zero_left = self._can_prune(t.left(n))
            is_zero = is_zero_left and is_zero_right
        self.is_zero[n] = is_zero
        return is_zero

    def is_root_zero(self):
        return self.is_zero[self.t.root()]

    # Copy the tree, skipping the prunable nodes
    def prune(self, n, tc, nc):
        t = self.t
        if t.is_internal(n):
            left, right = t.left(n), t.right(n)
            if not (self.is_zero[left] and self.is_zero[right]):
                s = t.get_split(n)
                tc.split(nc, s.feat_id, s.split_value)
                self.prune(right, tc, tc.right(nc))
                self.prune(left, tc, tc.left(nc))
            else:
                tc.set_leaf_value(nc, 0, 0.0)
        else:
            lv = t.get_leaf_value(n, 0)
            tc.set_leaf_value(nc, 0, lv)
