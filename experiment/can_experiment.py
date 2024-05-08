import os
import json
import click
import veritas
import numpy as np
import time

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
@click.option("--fold", default=0)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
def compress_cmd(dname, model_type, linclf_type, fold, seed, silent):
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, linclf_type, fold, silent)
    model_class = d.get_model_class(model_type)

    # if `linclf_type == Lasso`, the classification task is converted into a
    # regression task with -1 for the negative class and +1 for the positive
    # class. If `linclf_type == LogisticRegression`, it is normal binary
    # classification.

    key = util.get_key(model_type, linclf_type, seed)
    train_results = util.load_train_results()[key][dname][fold]
    train_results = [p for p in train_results if p["on_pareto_front"]]

    refine_results = []
    for tres in train_results:
        params = tres["params"]

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
        ## DO THE LEAF REFINEMENT HERE
        ## The resulting at is called `at_refined`
        at_refined = at_orig.copy()
        refine_time = time.time() - refine_time

        compress_result = {
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

        refine_results.append(compress_result)

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


if __name__ == "__main__":
    cli()
