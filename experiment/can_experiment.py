import os
import json
import click
import veritas
import numpy as np
import time
from sklearn.metrics import zero_one_loss
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression


import util
import tree_compress

@click.group()
def cli():
    pass

@cli.command("leaf_refine")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--penalty", type=click.Choice(["l1", "l2"]),
              default="l2")
@click.option("--fold", default=0)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
def leaf_refine_cmd(dname, model_type, linclf_type, penalty,fold, seed, silent):
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, linclf_type, fold, silent)
    model_class = d.get_model_class(model_type)

    # if `linclf_type == Lasso`, the classification task is converted into a
    # regression task with -1 for the negative class and +1 for the positive
    # class. If `linclf_type == LogisticRegression`, it is normal binary
    # classification.

    key = util.get_key(model_type, linclf_type, seed)
    train_results = util.load_train_results()[key][dname]

    refine_results = []
    for params_hash, folds in train_results.items():
        tres = folds[fold]
        params = tres["params"]

        if not tres["on_any_pareto_front"]:
            continue

        # Retrain the model
        clf, train_time = dtrain.train(model_class, params)
        mtrain = dtrain.metric(clf)
        mvalid = dvalid.metric(clf)
        mtest =  dtest.metric(clf)
        at_orig = veritas.get_addtree(clf, silent=True)

        assert np.abs(mtrain - tres["mtrain"]) < 1e-5
        assert np.abs(mvalid - tres["mvalid"]) < 1e-5
        assert np.abs(mtest - tres["mtest"]) < 1e-5


        refine_time = time.time()
        if penalty == "l2":
            refiner = LogisticRegression(penalty="l2", fit_intercept=True,
                                            solver="liblinear", 
                                            dual=True,
                                            max_iter=10000,
                                            warm_start=False)
        else:
            refiner = LogisticRegression(penalty="l1", fit_intercept=True,
                                solver="liblinear", 
                                max_iter=3000,
                                warm_start=False)
            
        alpha_list = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        sparse_train_x = transform_data_sparse(at_orig,dtrain.X) 
        sparse_valid_x = transform_data_sparse(at_orig,dvalid.X)
        best_score = float('inf')
        for alpha in  alpha_list:
            refiner.set_params(C = 1/alpha)
            refiner.fit(sparse_train_x,dtrain.y)
            score = zero_one_loss(dvalid.y, refiner.predict(sparse_valid_x))
            if score < best_score:
                best_score = score
                best_alpha = alpha
        refiner.set_params(C = 1/best_alpha)
        refiner.fit(sparse_train_x,dtrain.y)
        
        at_refined = at_orig.copy()
        at_refined = set_new_leaf_vals(at_refined, refiner.intercept_,refiner.coef_[0])
        refine_time = time.time() - refine_time

        refine_result = {
            "params": params,

            # Performance of the compressed model
            "compr_time": refine_time,
            "mtrain": dtrain.metric(at_refined),
            "mvalid": dvalid.metric(at_refined),
            "mtest": dtest.metric(at_refined),
            "ntrees": len(at_refined),
            "nnodes": int(at_refined.num_nodes()),
            "nleafs": int(at_refined.num_leafs()),
            "nnzleafs": int(tree_compress.count_nnz_leafs(at_refined)),
            "max_depth": int(at_refined.max_depth()),
        }

        refine_results.append(refine_result)

    results = {
        # Experimental settings
        "cmd": "train",
        "date_time": util.nowstr(),
        "hostname": os.uname()[1],
        "dname": dname,
        "model_type": model_type,
        "fold": fold,
        "linclf_type": linclf_type,
        "seed": seed,
        "metric_name": d.metric_name,

        # Any additional parameters
        #<param>: <value>

        # Result for all the hyper parameter settings
        "models": refine_results,
    
        # Data characteristics
        "ntrain": dtrain.X.shape[0],
        "nvalid": dvalid.X.shape[0],
        "ntest": dtest.X.shape[0],
    }
    if not silent:
        __import__('pprint').pprint(results)
    print(json.dumps(results))



def transform_data_sparse(at, x):
    row_ind, col_ind = [], []
    num_rows = x.shape[0]
    num_cols = 0

    for tree_index, t in enumerate(at):
        leaf_ids = t.get_leaf_ids()
        num_leaves = len(leaf_ids)
        offset = num_cols
        num_cols += num_leaves

        leaf2index = np.zeros(t.num_nodes(), dtype=int)
        for cnt, id in enumerate(leaf_ids):
            leaf2index[id] = cnt

        row_ind += list(range(num_rows))
        col_ind += list(map(lambda u: offset + leaf2index[u], t.eval_node(x)))

    return csr_matrix((np.ones(len(row_ind), dtype=np.float64),
                              (row_ind, col_ind)), shape=(num_rows, num_cols))

def set_new_leaf_vals(at, base_score, new_leaf_vals):
    at.set_base_score(0,base_score)
    offset = 0
    for tree_index, t in enumerate(at):
        for count , leaf_id in enumerate(t.get_leaf_ids()):
            t.set_leaf_value(leaf_id, 0, new_leaf_vals[offset + count])
        offset += count + 1
    return at
                              

if __name__ == "__main__":
    cli()
