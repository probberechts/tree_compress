import os
import json
import time
import click
import veritas

import prada
import model_params
import util
import tree_compress

from dataclasses import dataclass

from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error

SEED = 5823
TIMEOUT = 60
     
@click.group()
def cli():
    pass

@cli.command("list")
@click.option("--cmd", type=click.Choice(["train", "compress"]))
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--paramset", type=click.Choice(["full", "small"]),
              default="full")
@click.option("--abserr", default=0.01)
@click.option("--seed", default=SEED)
def print_configs(cmd, linclf_type, paramset, abserr, seed):
    if cmd == "train":
        for dname in util.DNAMES:
            d = prada.get_dataset(dname, seed=seed, silent=True)

            for model_type in ["xgb"]:#, "dt"]:
                folds = [i for i in range(util.NFOLDS)]

                grid = d.paramgrid(fold=folds)

                for cli_param in grid:
                    print("python experiment.py train",
                          dname,
                          "--model_type", model_type,
                          "--linclf_type", linclf_type,
                          "--paramset", paramset,
                          "--fold", cli_param["fold"],
                          "--seed", seed,
                          "--silent")
    elif cmd == "compress":
        for dname in util.DNAMES:
            d = prada.get_dataset(dname, seed=seed, silent=True)

            for model_type in ["xgb"]:#, "dt"]:
                folds = [i for i in range(util.NFOLDS)]

                grid = d.paramgrid(fold=folds)

                for cli_param in grid:
                    print("python experiment.py compress",
                          dname,
                          "--model_type", model_type,
                          "--linclf_type", linclf_type,
                          "--fold", cli_param["fold"],
                          "--abserr", abserr,
                          "--seed", seed,
                          "--silent")
    else:
        raise RuntimeError("dont know")

@cli.command("process_hyperparams")
def hyperparam_cmd():
    hyperparams = util.read_hyperparams()
    util.write_hyperparams(hyperparams)


@cli.command("process_results")
def result_cmd():
    organized = util.read_results()
    for key in organized.keys():
        for method_name in organized[key].keys():
            print("key", key, "method_name", method_name)
            print(organized[key][method_name].to_string())
    util.write_results(organized)


@dataclass
class HyperParamResult:
    clf: object
    train_time: float
    params: dict
    mtrain: float
    mvalid: float
    mtest: float


@cli.command("train")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--paramset", type=click.Choice(["full", "small"]),
              default="full")
@click.option("--fold", default=0)
@click.option("--abserr", default=0.01)
@click.option("--seed", default=SEED)
@click.option("--silent", is_flag=True, default=False)
def train_cmd(dname, model_type, linclf_type, paramset, fold, abserr, seed, silent):
    d = prada.get_dataset(dname, seed=seed, silent=silent)
    d.load_dataset()
    d.robust_normalize()
    #d.transform_target()
    d.scale_target()
    d.astype(veritas.FloatT)

    if d.is_regression():
        assert linclf_type == "Lasso"
    elif linclf_type == "Lasso":  # binaryclf as a regr problem
        d = d.as_regression_problem()
    else:
        linclf_type = "LogisticRegression"

    model_class = d.get_model_class(model_type)
    d.use_balanced_accuracy()
    dtrain, dtest = d.train_and_test_fold(fold, nfolds=util.NFOLDS)
    dtrain, dvalid = dtrain.split(0, nfolds=util.NFOLDS-1)

    if not silent:
        print("data size", d.X.shape)
        print("y:", d.y.mean(), "±", d.y.std())
        print("model_class", model_class)
        print("linclf_type", linclf_type)
        print("dtrain", dtrain.X.shape)
        print("dvalid", dvalid.X.shape)
        print("dtest", dtest.X.shape)

    hyperparam_time = time.time()
    if paramset == "full":
        param_dict = model_params.get_params(d, model_type)
    else:
        # Find the model size of the compressed model
        organized = util.get_results()
        key = util.get_key(model_type, linclf_type, seed)
        method_name = util.get_method_name("compress", model_type, "", abserr)

        print(organized[method_name][key].loc[dname])
        ntrees_ref, nnodes_ref, nnzleafs_ref = organized[method_name][key].loc[
            (dname, fold), ["ntrees", "nnodes", "nnzleafs"]
        ]
        mtestref = organized[method_name][key].loc[(dname, fold), "mtest"]
        param_dict = model_params.get_params_small(d, model_type, ntrees_ref, nnodes_ref, nnzleafs_ref)
    models = []

    for i, params in enumerate(d.paramgrid(**param_dict)):
        #__import__('pprint').pprint(params)
        clf, train_time = dtrain.train(model_class, params)
        mtrain = dtrain.metric(clf)
        mtest  = dtest.metric(clf)
        mvalid = dvalid.metric(clf)
        #print(f"{i:<3d} tr {mtrain:.3f}, va {mvalid:.3f}, te {mtest:.3f}", params)
        res = HyperParamResult(clf, train_time, params, mtrain, mvalid, mtest)
        models.append(res)

        try:
            at = veritas.get_addtree(res.clf, silent=True)
            print("run",
                  f"mtest {mtest:.3f} vs {mtestref:.3f} ({mtest >= mtestref:1d}),",
                  f"#nodes {at.num_nodes():5.0f} vs {nnodes_ref:5.0f}",
                  f"#nnz {tree_compress.count_nnz_leafs(at):5.0f} vs {nnzleafs_ref:5.0f}",
                  f"{mtest >= mtestref and nnodes_ref >= at.num_nodes():1d}")
        except IndexError:
            print("no trees?")
            print(params)

        del mtrain, mvalid, mtest
        del res, clf, train_time, params
    hyperparam_time = time.time() - hyperparam_time

    if paramset == "full":
        def model_filter(res):
            return True
    elif paramset == "small":
        def model_filter(res):
            at = veritas.get_addtree(res.clf, silent=True)
            return tree_compress.count_nnz_leafs(at) <= nnzleafs_ref
            #return at.num_nodes() <= nnodes
    best = max(filter(model_filter, models), key=lambda res: res.mvalid)
    at = veritas.get_addtree(best.clf, silent=True)

    if paramset == "small":
        print("final",
              f"mtest {best.mtest:.3f} vs {mtestref:.3f} ({best.mtest >= mtestref:1d}),",
              f"#nodes {at.num_nodes():5.0f} vs {nnodes_ref:5.0f}",
              f"nnz {tree_compress.count_nnz_leafs(at):5.0f} vs {nnzleafs_ref:5.0f},",
              f"{best.mtest >= mtestref and nnodes_ref >= at.num_nodes():1d}")

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

        # Settings for training
        "abserr": abserr,
        "paramset": paramset,
        "params": best.params,
        "metric_name": d.metric_name,

        # Model stats, for xgb,dt,... model here
        "best_train_time": float(best.train_time),
        "hyperparam_time": hyperparam_time,
        "time": hyperparam_time,

        "mtrain": float(best.mtrain),
        "mvalid": float(best.mvalid),
        "mtest": float(best.mtest),
        "ntrees": len(at),
        "nnodes": int(at.num_nodes()),
        "nnzleafs": int(tree_compress.count_nnz_leafs(at)),
        "max_depth": int(at.max_depth()),

        # Data characteristics
        "ntrain": dtrain.X.shape[0],
        "nvalid": dvalid.X.shape[0],
        "ntest": dtest.X.shape[0],
    }
    if not silent:
        __import__('pprint').pprint(results)
    print(json.dumps(results))

@cli.command("compress")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--fold", default=0)
@click.option("--abserr", default=0.01)
@click.option("--seed", default=SEED)
@click.option("--silent", is_flag=True, default=False)
@click.option("--plot", is_flag=True, default=False)
def compress_cmd(dname, model_type, linclf_type, fold, abserr, seed, silent, plot):
    d = prada.get_dataset(dname, seed=seed, silent=silent)
    d.load_dataset()
    d.robust_normalize()
    #d.transform_target()
    d.scale_target()
    d.astype(veritas.FloatT)

    all_params = util.get_hyperparams()
    params = all_params[util.get_key(model_type, linclf_type, seed)]
    params = params[dname]
    params = params[fold]

    if d.is_regression():
        assert linclf_type == "Lasso"
        def mymetric(ytrue, ypred):
            return -root_mean_squared_error(ytrue, ypred)
        raise RuntimeError("does not work yet")
    elif linclf_type == "Lasso":  # binaryclf as a regr problem
        d = d.as_regression_problem()
        def mymetric(ytrue, ypred):
            return balanced_accuracy_score(ytrue > 0.0, ypred > 0.0)
    else:
        linclf_type = "LogisticRegression"
        def mymetric(ytrue, ypred):
            return balanced_accuracy_score(ytrue, ypred)

    model_class = d.get_model_class(model_type)
    d.use_balanced_accuracy()
    dtrain, dtest = d.train_and_test_fold(fold, nfolds=util.NFOLDS)
    dtrain, dvalid = dtrain.split(0, nfolds=util.NFOLDS-1)

    if not silent:
        print("data size", d.X.shape)
        print("y:", d.y.mean(), "±", d.y.std())
        print("model_class", model_class)
        print("linclf_type", linclf_type)
        print("params:", params)
        print("dtrain", dtrain.X.shape)
        print("dvalid", dvalid.X.shape)
        print("dtest", dtest.X.shape)

    # Retrain the models with the best hyper-params found in the `train` experiment
    clf, train_time = dtrain.train(model_class, params)
    mtrain = dtrain.metric(clf)
    mvalid = dvalid.metric(clf)
    mtest = dtest.metric(clf)
    at = veritas.get_addtree(clf, silent=True)

    if not silent:
        print(f"{model_type} {d.metric_name}:",
              f"mtr {mtrain:.3f} mte {mtest:.3f} mva {mvalid:.3f}",
              f"in {train_time:.2f}s")

    data = tree_compress.Data(
            dtrain.X.to_numpy(), dtrain.y.to_numpy(),
            dtest.X.to_numpy(), dtest.y.to_numpy(),
            dvalid.X.to_numpy(), dvalid.y.to_numpy())

    compr = tree_compress.LassoCompress(
        data,
        at,
        metric=mymetric,
        isworse=lambda v, ref: ref-v > abserr,
        linclf_type=linclf_type,
        seed=seed,
        silent=silent
    )
    compr.no_convergence_warning = True
    compr_time = time.time()
    max_rounds = 2
    at_compr = compr.compress(max_rounds=max_rounds)
    compr_time = time.time() - compr_time
    #compr.compress(max_rounds=1)
    #compr.linclf_type = "LogisticRegression"
    #at_compr = compr.compress(max_rounds=1)
    record = compr.records[-1]

    if not silent:
        if at_compr.num_nodes() < 50:
            for i, t in zip(range(3), at_compr):
                print(t)
    
    if not silent:
        print(f"num_nodes {at.num_nodes()}->{at_compr.num_nodes()},",
              f"len {len(at)}->{len(at_compr)}",
              f"nnz_leafs {compr.records[0].nnz_leafs}->{record.nnz_leafs}")
        print()

    ntrees_rec = [int(r.ntrees) for r in compr.records]
    nnodes_rec = [int(r.nnodes) for r in compr.records]
    nnzleafs_rec = [int(r.nnz_leafs) for r in compr.records]
    mtrain_rec = [float(r.mtrain) for r in compr.records]
    mvalid_rec = [float(r.mvalid) for r in compr.records]
    mtest_rec = [float(r.mtest) for r in compr.records]

    results = {

        # Experimental settings
        "cmd": "compress",
        "date_time": util.nowstr(),
        "hostname": os.uname()[1],
        "dname": dname,
        "model_type": model_type,
        "fold": fold,
        "linclf_type": linclf_type,
        "seed": seed,

        # Compression params
        "abserr": abserr,
        "max_rounds": max_rounds,
        "alpha_search_steps": compr.alpha_search_round_nsteps,
        "tol": compr.tol,
        "warm_start": compr.warm_start,
        "clf_mtrain": mtrain,
        "clf_mvalid": mvalid,
        "clf_mtest": mtest,
        "params": params,

        # Model stats, for compressed model here
        "best_train_time": train_time,
        "compr_time": compr_time,
        "time": compr_time,

        "mtrain": float(record.mtrain),
        "mvalid": float(record.mvalid),
        "mtest": float(record.mtest),
        "ntrees": int(record.ntrees),
        "nnodes": int(record.nnodes),
        "nnzleafs": int(record.nnz_leafs),
        "max_depth": int(at.max_depth()),

        # Compress specific: stats of compression process
        "ntrees_rec": ntrees_rec,
        "nnodes_rec": nnodes_rec,
        "nnzleafs_rec": nnzleafs_rec,
        "metric_name": d.metric_name,
        "mtrain_rec": mtrain_rec,
        "mvalid_rec": mvalid_rec,
        "mtest_rec": mtest_rec,

        # Data characteristics
        "ntrain": dtrain.X.shape[0],
        "nvalid": dvalid.X.shape[0],
        "ntest": dtest.X.shape[0],
    }
    if not silent:
        __import__('pprint').pprint(results)
    print(json.dumps(results))

    if plot:
        import matplotlib.pyplot as plt

        mname = d.metric_name
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        ax0.semilogy(nnodes_rec, label="#nodes")
        ax0.semilogy(nnzleafs_rec, label="#non-zero leaves")
        ax0.legend()
        ax0.set_ylabel("num nodes/leaves")
        ax1.plot(mtrain_rec, label=f"{mname} train")
        ax1.plot(mtest_rec, label=f"{mname} test")
        ax1.plot(mvalid_rec, label=f"{mname} valid")
        ax1.axhline(y=mtest_rec[0], ls=":", color="gray")
        ax1.axhline(y=mtest_rec[0] - abserr,
                    ls="-.", color="gray")
        ax1.legend()
        ax1.set_xlabel("iteration")
        ax1.set_ylabel(mname)
        fig.suptitle(dname)

        #config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
        #config.stop_when_optimal = False
        #search = config.get_search(at_compr)
        #while search.steps(1000) != veritas.StopReason.NO_MORE_OPEN \
        #        and search.time_since_start() < 5.0:
        #    pass
        #    #print("Veritas", search.num_solutions(), search.num_open())
        #print("Veritas", search.num_solutions(), search.num_open())

        plt.show()



if __name__ == "__main__":
    cli()
