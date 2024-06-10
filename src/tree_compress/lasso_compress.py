import time
import numpy as np
import veritas
import itertools

from dataclasses import dataclass

from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, ElasticNet
from sklearn.preprocessing import OneHotEncoder

from .util import count_nnz_leafs, print_metrics, print_fit


@dataclass(init=False)
class CompressRecord:
    level: int
    tmapping: float
    ttransform: float
    tsearch: float

    at: veritas.AddTree

    ntrees: int
    nnodes: int
    nleafs: int
    nnzleafs: int

    mtrain: float
    mtest: float
    mvalid: float

    alpha: float
    clf_mtrain: float
    clf_mvalid: float

    def __init__(self, level, at):
        self.level = level
        self.tmapping = 0.0
        self.ttransform = 0.0
        self.tsearch = 0.0

        self.at = at

        self.ntrees = len(at)
        self.nnodes = at.num_nodes()
        self.nleafs = at.num_leafs()
        self.nnzleafs = count_nnz_leafs(at)

        self.mtrain = 0.0
        self.mtest = 0.0
        self.mvalid = 0.0

        self.alphas = np.nan
        self.clf_mtrain = np.nan
        self.clf_mvalid = np.nan


@dataclass(init=False)
class AlphaRecord:
    lo: float
    hi: float
    alpha: float

    clf_mtrain: float
    clf_mvalid: float
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
        self.clf_mtrain = 0.0
        self.clf_mvalid = 0.0
        self.num_params = 0
        self.num_removed = 0
        self.num_kept = 0
        self.frac_removed = 0.0
        self.fit_time = 0.0
        self.intercept = None
        self.coefs = None


class AlphaSearch:
    def __init__(self, round_nsteps, mtrain_ref, mvalid_ref, isworse_fun):
        """Search for a regularizion strength parameter.
        """
        self.round_nsteps = round_nsteps
        self.mtrain_ref = mtrain_ref
        self.mvalid_ref = mvalid_ref
        self.isworse_fun = isworse_fun

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
            self.lo, self.hi = self.next_lohis()
            self.set_lohis()
            self.step = 1

        lo, mid, hi = self.lohis[self.step - 1 : self.step + 2]
        alpha = np.power(10.0, mid)

        record = AlphaRecord(lo, hi, alpha)
        self.records.append(record)

        self.step += 1
        return record

    def isworse_tr(self, mtrain):
        return self.isworse_fun(mtrain, self.mtrain_ref)

    def isworse_va(self, mvalid):
        return self.isworse_fun(mvalid, self.mvalid_ref)

    def isnotworse_tr(self, mtrain):
        return not self.isworse_fun(mtrain, self.mtrain_ref)

    def isnotworse_va(self, mvalid):
        return not self.isworse_fun(mvalid, self.mvalid_ref)

    def overfits(self, mtrain, mvalid):
        cond1 = self.isnotworse_tr(mtrain)
        cond2 = self.isworse_va(mvalid)
        cond3 = self.isworse_fun(mvalid, mtrain)
        return cond1 and cond2 and cond3

    def underfits(self, mtrain, mvalid):
        cond1 = self.isworse_tr(mtrain)
        cond2 = self.isworse_va(mvalid)
        return cond1 and cond2

    def nsteps(self):
        return self.round_nsteps[self.round]

    def set_lohis(self):
        nsteps = self.nsteps()
        self.lohis = np.linspace(self.lo, self.hi, nsteps + 2)

    def quality_filter(self, records):
        filt = filter(
            lambda r: self.isnotworse_va(r.clf_mvalid)
            and r.frac_removed < 1.0
            and r.num_kept > 1
            and not self.isworse_fun(r.clf_mtrain, r.clf_mvalid),  # overfitting
            records,
        )
        return filt

    def next_lohis(self):
        nsteps = self.nsteps()
        num_rounds = len(self.round_nsteps)

        self.round += 1
        if self.round >= num_rounds:
            raise StopIteration()

        prev_round_records = self.records[-1 * nsteps :]

        # Out of the records whose validation metric is good enough...
        filt = self.quality_filter(prev_round_records)
        # ... pick the one with the highest alpha
        best = max(filt, default=None, key=lambda r: r.alpha)

        # If nothing was good enough, find the last record that was overfitting and the
        # first that was underfitting, and look at that transition in more detail
        r_over, r_under = None, None
        if best is None:
            for r in prev_round_records:
                if self.overfits(r.clf_mtrain, r.clf_mvalid):
                    r_over = r
                elif r_over is not None and self.underfits(r.clf_mtrain, r.clf_mvalid):
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
        m = max(self.quality_filter(self.records), default=None, key=lambda r: r.frac_removed)
        if m is None:
            return None
        allm = [r for r in self.quality_filter(self.records) if r.frac_removed == m.frac_removed]
        return allm[-1]


def _at_isregr(at):
    """ Check if given veritas.AddTree is regressor. """
    return at.get_type() in {
        veritas.AddTreeType.REGR,
        veritas.AddTreeType.REGR_MEAN,
    }

def _at_predlab(at, x):
    """ Predict hard labels for veritas.AddTree """
    if _at_isregr(at):
        return at.predict(x)
    elif at.num_leaf_values() == 1:
        return (at.eval(x)[:, 0] >= 0.0).astype(int)
    else:
        return at.eval(x).argmax(axis=1)

def _lasso_predlab(isregr, nlv, clf, x):
    """ Predict hard label for Lasso classifier """
    if isregr:
        return clf.predict(x)
    pred = clf.predict(x)
    if nlv > 1:
        nexamples = x.shape[0] // nlv
        pred = pred.reshape(nexamples, nlv, order="C")
        return pred.argmax(axis=1)
    else:
        return (pred >= 0.0).astype(int)



class Compress:
    def __init__(
        self,
        data,
        at,
        metric,  # e.g. rmse(ytrue, ypred) (for clf: labels, not weights)
        isworse,  # isworse(value, reference)
                  # e.g. relative error is more than 0.01, abs. error >= 2%
        alpha_search_round_nsteps=[8, 4, 4],
        linclf_type="Lasso",
        seed=988569,
        silent=False,
    ):
        self.d = data
        self.at = at
        self.nlv = at.num_leaf_values()
        self.metric = metric
        self.isworse = isworse
        self.alpha_search_round_nsteps = alpha_search_round_nsteps
        self.seed = seed
        self.silent = silent
        self.no_convergence_warning = silent
        self.tol = 1e-5
        self.linclf_type = linclf_type
        self.warm_start = True

        self.mtrain = self.metric(self.d.ytrain, _at_predlab(at, self.d.xtrain))
        self.mtest = self.metric(self.d.ytest, _at_predlab(at, self.d.xtest))
        self.mvalid = self.metric(self.d.yvalid, _at_predlab(at, self.d.xvalid))

        if self.is_regression():
            self.ytrain = self.d.ytrain
            self.ytest = self.d.ytest
            self.yvalid = self.d.yvalid
        else:
            self.y_encoder = OneHotEncoder(
                drop="if_binary", sparse_output=False, dtype=np.float32
            ).fit(self.d.ytrain.reshape(-1, 1))

            def trsf(y):
                ymat = self.y_encoder.transform(y) * 2.0 - 1.0
                return ymat.flatten(order="C")

            self.ytrain = trsf(self.d.ytrain.reshape(-1, 1))
            self.ytest = trsf(self.d.ytest.reshape(-1, 1))
            self.yvalid = trsf(self.d.yvalid.reshape(-1, 1))

        start_record = CompressRecord(-1, at)
        start_record.mtrain = self.mtrain
        start_record.mtest = self.mtest
        start_record.mvalid = self.mvalid
        self.records = [start_record]

        if not self.silent:
            print(
                "MODEL PERF:",
                f"mtr {self.mtrain:.3f} mte {self.mtest:.3f} mva {self.mvalid:.3f}",
            )

    def is_regression(self):
        return _at_isregr(self.at)

    def compress(self, *, max_rounds=2):
        last_record = self.records[-1]
        for i in range(max_rounds):
            if not self.silent:
                print(f"\nROUND {i+1}")

            self._compress_round()

            new_record = self.records[-1]
            has_improved = last_record.nnodes > new_record.nnodes
            last_record = new_record

            if not has_improved:
                break

        return last_record.at

    def _compress_round(self):
        for level in itertools.count():
            r = self._compress_level(level)

            if not self.silent:
                r0 = self.records[0]
                r1 = self.records[-1]
                print_metrics("orig", r0)
                print_metrics("prev", r1, rcmp=r0, cmp=self.isworse)
                print_metrics("now", r, rcmp=r0, cmp=self.isworse)

            self.records.append(r)

            max_depth = self.at.max_depth()
            if level >= max_depth:
                if not self.silent:
                    print(f"DONE, depth of tree reached {level}, {max_depth}")
                break

    def _compress_level(self, level):
        t = time.time()
        mapping, num_cols = self._get_matrix_mapping(self.at, level)
        tmapping = time.time() - t

        t = time.time()
        xxtrain = self._transformx(self.at, self.d.xtrain, mapping, num_cols)
        xxvalid = self._transformx(self.at, self.d.xvalid, mapping, num_cols)
        ttransform = time.time() - t

        if not self.silent:
            print(
                f"Level {level}, xxtrain.shape {xxtrain.shape},",
                f"mapping time: {tmapping:.2f}s,",
                f"transform time: {ttransform:.2f}s"
            )
        alpha_search = AlphaSearch(
            self.alpha_search_round_nsteps,
            self.mtrain,
            self.mvalid,
            self.isworse,
        )
        clf = self._get_linclf()

        tsearch = time.time()
        for alpha_record in alpha_search:
            self._update_clf(clf, alpha_record.alpha)
            clf = self._fit_coefficients(clf, xxtrain, xxvalid, alpha_record)

            #at = self._prune_trees(self.at, clf.intercept_, clf.coef_, mapping)
            #mtr = self.metric(self.d.ytrain, _at_predlab(at, self.d.xtrain))
            #mva = self.metric(self.d.yvalid, _at_predlab(at, self.d.xvalid))
            #print("check mtrain", mtr, np.round(mtr - alpha_record.clf_mtrain, 3))
            #eval_at = at.eval(self.d.xtrain)[:,0]
            #eval_clf = clf.predict(xxtrain)
            #diff = eval_at - eval_clf
            #print(eval_at.round(5))
            #print(eval_clf.round(5))
            #print(diff.round(5), diff[0], np.std(diff))
            #print("check mvalid", mva, np.round(mva - alpha_record.clf_mvalid, 3))
            #eval_at = at.eval(self.d.xvalid)[:,0]
            #eval_clf = clf.predict(xxvalid)
            #diff = eval_at - eval_clf
            #print(eval_at.round(5))
            #print(eval_clf.round(5))
            #print(diff.round(5), diff[0], np.std(diff))
            #print()

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
            self.at = self._prune_trees(self.at, intercept, coefs, mapping)

            best_alpha = best.alpha
            clf_mtrain = best.clf_mtrain
            clf_mvalid = best.clf_mvalid

        # record
        mtrain_compr = self.metric(self.d.ytrain, _at_predlab(self.at, self.d.xtrain))
        mtest_compr = self.metric(self.d.ytest, _at_predlab(self.at, self.d.xtest))
        mvalid_compr = self.metric(self.d.yvalid, _at_predlab(self.at, self.d.xvalid))
        if not self.silent:
            print("best check mtrain", mtrain_compr, clf_mtrain, mtrain_compr-clf_mtrain,
                  f"(alpha={best_alpha:.4f})")
            print("best check mvalid", mvalid_compr, clf_mvalid, mvalid_compr-clf_mvalid)
        record = CompressRecord(level, self.at)
        record.mtrain = mtrain_compr
        record.mtest = mtest_compr
        record.mvalid = mvalid_compr
        record.alpha = best_alpha
        record.clf_mtrain = clf_mtrain
        record.clf_mvalid = clf_mvalid
        record.tmapping = tmapping
        record.ttransform = ttransform
        record.tsearch = tsearch

        return record


    def is_single_target(self):
        return self.nlv == 1

    def _get_matrix_mapping(self, at, level):
        mapping = []  # tree_index -> node_idx -> col_idex
        if self.is_single_target():
            num_cols = 0  # we use intercept
        else:  # one 'intercept' per class/target
            num_cols = self.nlv

        for t in at:
            mapping0 = {}  # leaf_id -> node_id at level along root-to-leaf path
            mapping1 = {}  # node_id at level -> xxmatrix column index

            # leaf_id -> [nodes along root-to-leaf path]
            for leaf_id in t.get_leaf_ids():
                n = leaf_id
                path = [n]
                while not t.is_root(n):
                    n = t.parent(n)
                    path.insert(0, n)

                if len(path) == 1:  # a single leaf node tree is captured by intercept
                    assert t.is_leaf(t.root()) and t.is_root(leaf_id)
                    continue

                # if this leaf is higher up than the level, just take the leaf
                n_at_level = path[level] if len(path) > level else path[-1]
                mapping0[leaf_id] = n_at_level

                if n_at_level not in mapping1:
                    mapping1[n_at_level] = num_cols
                    if t.is_root(n_at_level):    # only multiplicative coefficients
                        num_cols += 1
                    elif t.is_leaf(n_at_level):  # only biases
                        num_cols += self.nlv
                    else:
                        num_cols += 1 + self.nlv

            mapping.append((mapping0, mapping1))
        return mapping, num_cols

    def _transformx(self, at, x, mapping, num_cols):
        num_rows = x.shape[0]
        xx = np.zeros((self.nlv * num_rows, num_cols), dtype=np.float32, order="F")

        numba_success = self._transformx_numba(at, x, xx, mapping)
        if numba_success:
            return xx

        raise RuntimeError("numba: no numba not impl")

        # AddTree transformation
        for t, (mapping0, mapping1) in zip(at, mapping):
            for i, leaf_id in enumerate(t.eval_node(x)):
                n_at_level = mapping0[leaf_id]
                col_idx = mapping1[n_at_level]
                if t.is_root(n_at_level) or not t.is_leaf(n_at_level):
                    for target in range(self.nlv):
                        leaf_value = t.get_leaf_value(leaf_id, target)
                        xx[i, col_idx+target] = leaf_value
                    col_idx += self.nlv
                if t.is_leaf(n_at_level) or not t.is_root(n_at_level):
                    xx[i, col_idx] = 1.0

        return xx

    def _transformx_numba(self, at, x, xx, mapping):
        try:
            import numba
        except ModuleNotFoundError:
            return False

        #t = time.time()
        xnode = np.array([t.eval_node(x) for t in at], dtype=np.int32)
        #txnode = time.time() - t

        #t = time.time()
        M = len(at)
        max_leaf_id = max(max(m[0].keys(), default=0) for m in mapping)
        n_at_levels = np.full((M, max_leaf_id + 1), -1, dtype=np.int32)
        leafvals = np.zeros((M, max_leaf_id + 1, self.nlv), dtype=np.float32)

        for m, tree, (mapping0, mapping1) in zip(range(M), at, mapping):
            for leaf_id, n_at_level in mapping0.items():
                n_at_levels[m, leaf_id] = n_at_level
                for target in range(self.nlv):
                    leafvals[m, leaf_id, target] = tree.get_leaf_value(leaf_id, target)

        #tn_at_levels = time.time() - t

        #t = time.time()
        max_n_at_level = max(max(m[1].keys(), default=0) for m in mapping)
        col_idxs = np.full((M, max_n_at_level + 1), -1, dtype=np.int32)
        node_types = np.zeros((M, max_n_at_level + 1), dtype=np.uint8)
        for m, tree, (mapping0, mapping1) in zip(range(M), at, mapping):
            for n_at_level, col_idx in mapping1.items():
                col_idxs[m, n_at_level] = col_idx
                is_root = np.uint8(tree.is_root(n_at_level) * 0b01)
                is_leaf = np.uint8(tree.is_leaf(n_at_level) * 0b10)
                node_types[m, n_at_level] = is_root | is_leaf 

        #tcol_idxs = time.time() - t

        @numba.jit(
            "(int32, float32[:,:], int32[:,:], float32[:,:,:], int32[:,:], int32[:,:], uint8[:,:])",
            nogil=True,
            parallel=False,
            boundscheck=True,
            cache=False,
        )
        def __transformx(nlv, xx, xnode, leafvals, n_at_levels, col_idxs, node_types):
            if nlv > 1:  # if not self.is_single_target():
                for i in range(xnode.shape[1]):  # iterate over examples
                    for target in range(nlv):
                        ii = i*nlv + target
                        xx[ii, target] = 1.0  # intercept for target

            for m in range(xnode.shape[0]):  # iterate over trees (could use prange)
                for i in range(xnode.shape[1]):  # iterate over examples
                    leaf_id = xnode[m, i]
                    n_at_level = n_at_levels[m, leaf_id]
                    col_idx = col_idxs[m, n_at_level]
                    node_type = node_types[m, n_at_level]

                    is_root = (node_type & 0b01) == 0b01
                    is_leaf = (node_type & 0b10) == 0b10

                    for target in range(nlv):
                        ii = i*nlv + target
                        leaf_value = leafvals[m, leaf_id, target]
                        if is_root and is_leaf:
                            continue
                        if is_root or not is_leaf:
                            xx[ii, col_idx] = leaf_value
                        if is_leaf or not is_root:
                            offset = int(is_root or not is_leaf)
                            xx[ii, col_idx+offset+target] = 1.0

        #t = time.time()
        __transformx(self.nlv, xx, xnode, leafvals, n_at_levels, col_idxs, node_types)
        #tnumba = time.time() - t

        #print("numba DEBUG", f"in {txnode:.2f}s {tn_at_levels:.2f}s {tcol_idxs:.2f}s {tnumba:.2f}s")

        return True

    def _get_linclf(self):
        self.seed += 1

        if self.linclf_type == "Lasso":
            return Lasso(
                fit_intercept=self.is_single_target(),
                alpha=1.0,
                random_state=self.seed,
                max_iter=10_000,
                tol=self.tol,
                precompute=True,
                selection="random",
                copy_X=False,
                warm_start=self.warm_start,
            )
        #elif self.linclf_type == "ElasticNet":
        #    return ElasticNet(
        #        fit_intercept=False,
        #        l1_ratio=0.9,
        #        random_state=self.seed,
        #        max_iter=10_000,
        #        tol=self.tol,
        #        warm_start=True,
        #        precompute=True,
        #        selection="cyclic",
        #        copy_X=False,
        #    )
        elif self.linclf_type == "LogisticRegression":
            return LogisticRegression(
                fit_intercept=self.is_single_target(),
                penalty="l1",
                C=1.0,
                solver="liblinear",
                max_iter=5_000,
                tol=self.tol,
                n_jobs=1,
                random_state=self.seed,
                warm_start=self.warm_start,
            )
        else:
            raise RuntimeError("linclf_type not Lasso or LogisticRegression")

    def _update_clf(self, clf, alpha):
        assert alpha > 0.0
        if isinstance(clf, Lasso) or isinstance(clf, ElasticNet):
            clf.alpha = 0.001 * alpha
        elif isinstance(clf, LogisticRegression):
            assert clf.penalty == "l1"
            clf.C = 1.0 / alpha
        else:
            raise RuntimeError("_update_clf ???")

    def _fit_coefficients(self, clf, xxtrain, xxvalid, alpha_record):
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        fit_time = time.time()
        if self.no_convergence_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                clf.fit(xxtrain, self.ytrain)
        else:
            clf.fit(xxtrain, self.ytrain)
        fit_time = time.time() - fit_time

        num_params = np.prod(clf.coef_.shape)
        num_removed = np.sum(np.abs(clf.coef_) < self.tol)
        frac_removed = num_removed / num_params

        isregr = self.is_regression()
        yhat_train = _lasso_predlab(isregr, self.nlv, clf, xxtrain)
        yhat_valid = _lasso_predlab(isregr, self.nlv, clf, xxvalid)

        #wrongs = np.nonzero((yhat_valid != self.d.yvalid))[0]
        #print(wrongs)
        #print("AT")
        #print(self.at.eval(self.d.xvalid[wrongs[0:4], :]).round(2))
        #print("PRED")
        #print(clf.predict(xxvalid[wrongs[0:4], :]).round(2))
        #print("LABELS")
        #print("  at", self.at.eval(self.d.xvalid[wrongs[0:4], :]).argmax(axis=1))
        #print(" clf", clf.predict(xxvalid[wrongs[0:4], :]).argmax(axis=1))
        #print("true", self.d.yvalid[wrongs[0:4]])

        alpha_record.clf_mtrain = self.metric(self.d.ytrain, yhat_train)
        alpha_record.clf_mvalid = self.metric(self.d.yvalid, yhat_valid)
        alpha_record.num_params = num_params
        alpha_record.num_removed = num_removed
        alpha_record.num_kept = num_params - num_removed
        alpha_record.frac_removed = frac_removed
        alpha_record.intercept = np.copy(clf.intercept_)
        alpha_record.coefs = np.copy(clf.coef_.ravel())
        alpha_record.fit_time = fit_time

        return clf

    def _per_tree_opt(self, num_trees):
        level = self.at.max_depth() + 1

        print("START PRE FIT")
        pre_tuned_at = veritas.AddTree(self.nlv, veritas.AddTreeType.CLF_MEAN)

        for m in range(0, len(self.at), num_trees):
            sub_at = veritas.AddTree(self.nlv, veritas.AddTreeType.REGR)
            mend = min(m+num_trees, len(self.at))
            for mm in range(m, mend):
                sub_at.add_tree(self.at[mm])
            mapping, num_cols = self._get_matrix_mapping(sub_at, level)
            xxtrain = self._transformx(sub_at, self.d.xtrain, mapping, num_cols)
            xxvalid = self._transformx(sub_at, self.d.xvalid, mapping, num_cols)

            clf = LinearRegression().fit(xxtrain, self.ytrain)
            print(f"ACC {m}:{mend}",
                  np.mean(clf.predict(xxtrain).argmax(axis=1) == self.d.ytrain),
                  np.mean(clf.predict(xxvalid).argmax(axis=1) == self.d.yvalid))

            sub_at_pruned = self._prune_trees(sub_at, clf.intercept_, clf.coef_, mapping)
            print(f"ACC {m}:{mend}",
                  np.mean(sub_at_pruned.predict(self.d.xtrain).argmax(axis=1) == self.d.ytrain),
                  np.mean(sub_at_pruned.predict(self.d.xvalid).argmax(axis=1) == self.d.yvalid))

            pre_tuned_at.add_trees(sub_at_pruned)

        print("END PRE FIT")

        self.at = pre_tuned_at
        print("ACC",
              np.mean(pre_tuned_at.predict(self.d.xtrain).argmax(axis=1) == self.d.ytrain),
              np.mean(pre_tuned_at.predict(self.d.xvalid).argmax(axis=1) == self.d.yvalid))

    def _prune_trees(self, at, intercept, coefs, mapping):
        addtree_type = veritas.AddTreeType.CLF_MEAN
        if self.is_regression():
            addtree_type = veritas.AddTreeType.REGR

        atp = veritas.AddTree(self.nlv, addtree_type)
        atpp = atp.copy()

        for t, (mapping0, mapping1) in zip(at, mapping):
            # special case: if coef of full tree is 0.0, then drop the tree
            if t.root() in mapping1:
                col_idx = mapping1[t.root()]
                coef = coefs[col_idx]
                if coef < self.tol:
                    continue

            self._copy_tree(t, atp.add_tree(), coefs, mapping1)

            tp = atp[len(atp) - 1]
            pruner = _TreeZeroLeafPruner(tp)
            if not pruner.is_root_zero():
                tpp = atpp.add_tree()
                pruner.prune(tp.root(), tpp, tpp.root())

        if self.is_single_target():
            if isinstance(intercept, np.ndarray) and intercept.ndim > 1:
                assert len(intercept) == 1
                base_score = intercept[0]
            else:
                base_score = intercept
            atpp.set_base_score(0, base_score)
        else:
            for target in range(self.nlv):
                base_score = coefs[target]
                atpp.set_base_score(target, base_score)
        return atpp

    def _copy_tree(self, t, tc, coefs, mapping1):
        self._copy_subtree(t, t.root(), tc, tc.root(), coefs, mapping1)

    def _copy_subtree(self, t, n, tc, nc, coefs, mapping1):
        #coefs[abs(coefs) < self.tol] = 0.0

        zero_bias = np.zeros(self.nlv)
        stack = [(n, nc, 1.0, zero_bias)]
        while len(stack) > 0:
            n, nc, coef, bias = stack.pop()

            if n in mapping1:
                col_idx = mapping1[n]
                if t.is_root(n):
                    coef = coefs[col_idx]
                    bias = zero_bias
                elif t.is_leaf(n):
                    coef = 0.0
                    bias = coefs[col_idx:col_idx+self.nlv]
                else:
                    coef = coefs[col_idx]
                    bias = coefs[col_idx+1:col_idx+self.nlv+1]

                if abs(coef) <= self.tol:  # skip the branch, just predict bias
                    #if not t.is_leaf(n) and np.all(abs(bias)) > self.tol:
                    #    print(f"cutting off branch {n} of size {t.tree_size(n)}, leaf value {bias.round(3)}")
                    for k in range(self.nlv):
                        tc.set_leaf_value(nc, k, bias[k])
                    continue

                #print(coef, "coef", nc)
                #print(bias, "bias", nc)

            if t.is_internal(n):
                s = t.get_split(n)
                tc.split(nc, s.feat_id, s.split_value)
                stack.append((t.right(n), tc.right(nc), coef, bias))
                stack.append((t.left(n), tc.left(nc), coef, bias))
            else:
                for target in range(self.nlv):
                    #leaf_value = bias[k] + np.dot(coef[k, :], leaf_values)
                    leaf_value = bias[target] + coef * t.get_leaf_value(n, target)
                    #print(f"new leaf value node {nc} target {target} {leaf_value}")
                    tc.set_leaf_value(nc, target, leaf_value)


class _TreeZeroLeafPruner:
    def __init__(self, t):
        self.t = t
        self.nlv = t.num_leaf_values()
        self.is_zero = np.zeros(t.num_nodes(), dtype=bool)
        self._can_prune(t.root())

    # Mark which subtrees all have leaf values equal, and can be pruned
    def _can_prune(self, n):
        t = self.t
        if t.is_leaf(n):
            is_zero = all(t.get_leaf_value(n, k) == 0.0 for k in range(self.nlv))
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
                pass  # leaf values are zero
        else:
            for k in range(self.nlv):
                lv = t.get_leaf_value(n, k)
                tc.set_leaf_value(nc, k, lv)
