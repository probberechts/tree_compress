import os
import json
import time
import click
import veritas

import prada
import model_params
import util

from datetime import datetime

from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error

SEED = 5823
NFOLDS = 5
TIMEOUT = 60
     
@click.group()
def cli():
    pass

DNAMES = [
    "Img[bin]",
    "Electricity",
    #"CovtypeNumeric",
    #"Covtype",
    "MagicTelescope",
    "BankMarketing",
    "Bioresponse",
    "MiniBooNE",
    "DefaultCreditCardClients",
    "EyeMovements",
    "Diabetes130US",
    "Jannis",
    "Heloc",
    "Credit",
    "California",
    "Albert",
    "CompasTwoYears",
    "RoadSafety",
    #"AtlasHiggs",
    "SantanderCustomerSatisfaction",
    #"BreastCancer",
    "Vehicle",
    "Spambase",
    "Phoneme",
    "Nomao",
    "Banknote",
    "Adult",
    "Ijcnn1",
    "Webspam",
    "Mnist[2v4]",
    "FashionMnist[2v4]",
    "Houses[bin]",
    "CpuAct[bin]",
    "MercedesBenzManufacturing[bin]",
    "BikeSharingDemand[bin]",
    "Yprop41[bin]",
    "Abalone[bin]",
    "DryBean[6vRest]",
    "Volkert[2v7]",
    "Seattlecrime6[bin]"
    #"HiggsBig",
    #"KddCup99",
]


@cli.command("list")
@click.option("--abserr", default=0.01)
@click.option("--seed", default=SEED)
def print_configs(abserr, seed):
    for dname in DNAMES:
        d = prada.get_dataset(dname, seed=seed, silent=True)

        for model_type in ["xgb"]:#, "dt"]:
            folds = [i for i in range(NFOLDS)]

            grid = d.paramgrid(fold=folds)

            for cli_param in grid:
                print("python experiment.py compress",
                      dname,
                      "--model_type", model_type,
                      "--fold", cli_param["fold"],
                      "--abserr", abserr,
                      "--lasso",
                      "--seed", seed,
                      "--silent")

@cli.command("results")
@click.option("--abserr", default=0.01)
@click.option("--seed", default=SEED)
def view_results(abserr, seed):
    organized = util.read_results(abserr, seed)
    util.get_essential_stats(organized)



                                       


@cli.command("compress")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--lasso", is_flag=True, default=False)
@click.option("--fold", default=0)
@click.option("--abserr", default=0.01)
@click.option("--seed", default=SEED)
@click.option("--silent", is_flag=True, default=False)
@click.option("--plot", is_flag=True, default=False)
def compress(dname, model_type, lasso, fold, abserr, seed, silent, plot):
    d = prada.get_dataset(dname, seed=seed, silent=silent)
    d.load_dataset()
    d.robust_normalize()
    #d.transform_target()
    d.scale_target()
    d.astype(veritas.FloatT)

    as_regression_problem = lasso
    if d.is_regression():
        linclf_type = "Lasso"
        def mymetric(ytrue, ypred):
            return -root_mean_squared_error(ytrue, ypred)
        raise RuntimeError("not working")
    elif as_regression_problem:  # binaryFalse clf as a regr problem
        linclf_type = "Lasso"
        d = d.as_regression_problem()
        def mymetric(ytrue, ypred):
            return balanced_accuracy_score(ytrue > 0.0, ypred > 0.0)
    else:
        linclf_type = "LogisticRegression"
        def mymetric(ytrue, ypred):
            return balanced_accuracy_score(ytrue, ypred)

    model_class = d.get_model_class(model_type)
    d.use_balanced_accuracy()

    if not silent:
        print("data size", d.X.shape)
        print("y:", d.y.mean(), "±", d.y.std())
        print("model_class", model_class)
        print("linclf_type", linclf_type)

    import tree_compress

    dtrain, dtest = d.train_and_test_fold(fold, nfolds=NFOLDS)
    dtrain, dvalid = dtrain.split(0, nfolds=NFOLDS-1)

    hyperparam_time = time.time()
    param_dict = model_params.get_params(d, model_type)
    models = []
    for i, params in enumerate(d.paramgrid(**param_dict)):
        #__import__('pprint').pprint(params)
        clf, train_time = dtrain.train(model_class, params)
        mtrain = dtrain.metric(clf)
        #mtest  = dtest.metric(clf)
        mvalid = dvalid.metric(clf)
        models.append((clf, train_time, params, mtrain, mvalid))
    hyperparam_time = time.time() - hyperparam_time

    clf, train_time, params, mtrain, mvalid = max(models, key=lambda m: m[-1])
    mtest = dtest.metric(clf)
    at = veritas.get_addtree(clf)

    del models

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
        seed=seed,
        silent=silent
    )
    compr.linclf_type = linclf_type
    compr.no_convergence_warning = True
    compr_time = time.time()
    max_rounds = 2
    at_compr = compr.compress(max_rounds=max_rounds)
    compr_time = time.time() - compr_time
    #compr.compress(max_rounds=1)
    #compr.linclf_type = "LogisticRegression"
    #at_compr = compr.compress(max_rounds=1)
    record = compr.records[-1]

    # try again:
    # python experiments.py test_idea --model_type xgb --abserr 0.01 Albert
    #   --> last level probably only works for alpha=0.0?
    # python experiments.py test_idea --model_type rf --abserr 0.02 EyeMovements
    #   --> becomes worse, worse, worse at each level
    # python experiments.py test_idea --model_type rf --abserr 0.02 VisualizingSoil
    #   --> all first 4 alphas are bad, so it picks the first one

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
    mtest_rec = [float(r.mtest) for r in compr.records]
    mvalid_rec = [float(r.mvalid) for r in compr.records]

    now = datetime.now() # current date and time

    results = {
        "date_time": now.strftime("%m/%d/%Y, %H:%M:%S"),
        "hostname": os.uname()[1],
        "dname": dname,
        "model_type": model_type,
        "params": params,
        "fold": fold,
        "abserr": abserr,
        "linclf_type": linclf_type,
        "max_rounds": max_rounds,
        "alpha_search_steps": compr.alpha_search_round_nsteps,
        "tol": compr.tol,
        "warm_start": compr.warm_start,
        "seed": seed,
        "compr_seed": compr.seed,
        "clf_mtrain": mtrain,
        "clf_mtest": mtest,
        "clf_mvalid": mtest,
        "compr_time": compr_time,
        "hyperparam_time": hyperparam_time,
        "clf_train_time": train_time,
        "ntrees_rec": ntrees_rec,
        "nnodes_rec": nnodes_rec,
        "nnzleafs_rec": nnzleafs_rec,
        "metric_name": d.metric_name,
        "mtrain_rec": mtrain_rec,
        "mtest_rec": mtest_rec,
        "mvalid_rec": mvalid_rec,
        "ntrain": dtrain.X.shape[0],
        "ntest": dtest.X.shape[0],
        "nvalid": dvalid.X.shape[0],
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
