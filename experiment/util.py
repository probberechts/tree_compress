import os
import joblib
import json
import colorama
import numpy as np
import pandas as pd

from datetime import datetime

LBRACE = "{"
RBRACE = "}"

NFOLDS = 5
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

def nowstr():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

def read_jsons():
    jsons = []
    for f in os.listdir("results/"):
        if not f.endswith(".txt"):
            continue
        f = os.path.join("results", f)
        if not os.path.isfile(f):
            continue
        with open(f) as fh:
            for line in fh.readlines():
                if not line.startswith(LBRACE):
                    continue

                j = json.loads(line)
                j["file"] = f
                jsons.append(j)
    return jsons

def get_or_insert(d, key, value_generator):
    if key in d:
        return d[key]
    else:
        d[key] = value_generator()
        return d[key]

def get_key(*args):
    if len(args) == 1:
        j = args[0]
        model_type = j["model_type"]
        linclf_type = j["linclf_type"]
        seed = j["seed"]
    else:
        model_type, linclf_type, seed = args
    return f"{model_type}-{linclf_type}-{seed}"

def read_hyperparams(jsons=None):
    if  jsons is None:
        jsons = read_jsons()

    params = {}
    for j in jsons:
        if j["task"] != "train":
            continue
        if j["paramset"] != "full":
            continue

        dname = j["dname"]
        fold = j["fold"]
        key = get_key(j)

        forkey = get_or_insert(params, key, lambda: {})
        fordname = get_or_insert(forkey, dname, lambda: {})
        fordname[fold] = j["params"]

    return params

def write_hyperparams(hyperparams):
    joblib.dump(hyperparams, "results/processed/hyperparams.joblib")


def get_hyperparams():
    return joblib.load("results/processed/hyperparams.joblib")


def get_method_name(*args):
    if len(args) == 1:
        j = args[0]
        cmd = j["cmd"]
        model_type = j["model_type"]
        paramset = j["paramset"] if cmd == "train" else ""
        abserr = j["abserr"] if cmd == "compress" else 0.01
    else:
        cmd, model_type, paramset, abserr = args
    suffix = f"compr{abserr*1000:04.0f}" if cmd == "compress" else paramset
    return f'{model_type}_{suffix}'


def read_results(jsons=None):
    if  jsons is None:
        jsons = read_jsons()

    # Organize by:
    # - dname
    # - seed
    # - abserr (0.005, 0.01, 0.02)
    # - fold (0, 1, ..., 4)
    # - model_type
    #    + xgb         rf         dt           J48 (WEKA)       
    #    + xgb.compr   rf.compr   dt.compr     J48.compr?   (per abserr)
    #    + xgb.small   rf.small   dt.small     J48.small
    #
    # Big dict:
    # [seed][model_type] = DataFrame
    #   with multi-index
    #     - dname x fold
    #   with columns
    #     - mtrain
    #     - mvalid
    #     - mtest
    #     - nnodes
    #     - nleaves
    #     - nnz_leaves
    #     - train / compress time
    #
    # compression specific columns
    #     - abserr
    #     - linclf_type

    organized = {}
    for j in jsons:
        dname = j["dname"]
        fold = j["fold"]
        key = get_key(j)
        method_name = get_method_name(j)

        index = pd.MultiIndex.from_product(
            [DNAMES, list(range(NFOLDS))], names=["dname", "fold"]
        )
        columns = ["mtrain", "mvalid", "mtest", "ntrees", "nnodes", "nnzleafs", "time"]
        formethod = get_or_insert(organized, method_name, lambda: {})
        df = get_or_insert(
            formethod, key, lambda: pd.DataFrame(np.nan, index=index, columns=columns)
        )

        for k in columns:
            try:
                df.loc[(dname, fold), k] = j[k]
            except KeyError:
                print("KEY ERROR", k, method_name, "in", j["file"])
                continue

    return organized

def write_results(organized):
    joblib.dump(organized, "results/processed/results.joblib")


def get_results():
    return joblib.load("results/processed/results.joblib")


def print_metrics(lab, m0, m1, abserr):
    RST = colorama.Style.RESET_ALL
    RED = colorama.Fore.RED
    GRN = colorama.Fore.GREEN

    s = " " * len(lab)
    m0m = np.mean(m0)
    m1m = np.mean(m1)
    m0s = np.std(m0)
    m1s = np.std(m1)
    diff = m1m-m0m
    clr = RED if diff > abserr else GRN
    print(lab, " ".join(f"{x:.3f}" for x in m0), "|", f"{m0m:.3f}±{m0s:.3f}")
    print(s,   " ".join(f"{x:.3f}" for x in m1), "|", f"{m1m:.3f}±{m1s:.3f}", f"{clr}{diff:.3f}{RST}")

def print_nnzleafs(lab, nnz0, nnz1):
    s = " " * len(lab)
    m0m = np.mean(nnz0)
    m1m = np.mean(nnz1)
    m0s = np.std(nnz0)
    m1s = np.std(nnz1)
    m1r = [x/y for x, y in zip(nnz0, nnz1)]
    m1rm = np.mean(m1r)
    m1rs = np.std(m1r)
    print(lab, " ".join(f"{x:5.0f}" for x in nnz0), "|", f"{m0m:5.0f}±{m0s:<5.0f}")
    print(s,   " ".join(f"{x:5.0f}" for x in nnz1), "|", f"{m1m:5.0f}±{m1s:<5.0f}")
    print(s,   " ".join(f"{r:4.0f}×" for r in m1r), "|", f"{m1rm:5.0f}±{m1rs:<5.0f}")

def get_essential_stats(organized):
    columns = ["n", "bal.acc.train", "bal.acc.test", "bal.acc.test loss", "nnz.leafs", "nnz.fraction", "time"]
    dnames = [n[0:15] for n in organized.keys()]
    index = pd.MultiIndex.from_product([organized.keys(), ["xgb", "xgb.compr", "dt", "dt.compr"]])

    df = pd.DataFrame(" ", columns=columns, index=index)

    columns_raw = [
        "bal.acc.train.mean",
        "bal.acc.train.std",
        "bal.acc.test.mean",
        "bal.acc.test.std",
        "nnz.leafs",
        "time",
    ]
    dfraw = pd.DataFrame("-", columns=columns_raw, index=index)

    nnz_fractions = []

    for dname, fordata in organized.items():
        n = len(fordata["xgb"])
        df.loc[(dname, "xgb"), "n"] = n
        tny = "\\textcolor{gray}{\\scriptsize " #}
        for model_type, formodel in fordata.items():
            print(f"{dname} {model_type}")
            for k0, k1 in [("mtrain_rec", "bal.acc.train"), ("mtest_rec", "bal.acc.test")]:
                m0 = np.array([j[k0][0]*100.0 for j in formodel.values()])
                m1 = np.array([j[k0][-1]*100.0 for j in formodel.values()])
                df.loc[(dname, model_type), k1] = f"{m0.mean():5.1f}\\% {tny}±{m0.std():<4.1f}}}"
                df.loc[(dname, f"{model_type}.compr"), k1] = f"{m1.mean():5.1f}\\% {tny}±{m1.std():<4.1f}}}"
                if "test" in k1:
                    df.loc[(dname, f"{model_type}.compr"), "bal.acc.test loss"] = (
                        f"{np.mean(m1-m0):+4.1f}\\%"
                    )
                dfraw.loc[(dname, model_type), f"{k1}.mean"] = np.mean(m0)
                dfraw.loc[(dname, model_type), f"{k1}.std"] = np.std(m0)
                dfraw.loc[(dname, f"{model_type}.compr"), f"{k1}.mean"] = np.mean(m1)
                dfraw.loc[(dname, f"{model_type}.compr"), f"{k1}.std"] = np.std(m1)

                print_metrics(k0[0:3], m0, m1, 0.01)

            nnzleaf0 = np.array([j["nnzleafs_rec"][0] for j in formodel.values()])
            nnzleaf1 = np.array([j["nnzleafs_rec"][-1] for j in formodel.values()])
            nnzleaf_frac = nnzleaf0 / nnzleaf1

            nnz_fractions += list(nnzleaf_frac)

            df.loc[(dname, model_type), "nnz.leafs"] = f"{nnzleaf0.mean():4.0f} {tny}±{nnzleaf0.std():<4.0f}}}"
            df.loc[(dname, f"{model_type}.compr"), "nnz.leafs"] = f"{nnzleaf1.mean():4.0f} {tny}±{nnzleaf1.std():<4.0f}}}"
            df.loc[(dname, f"{model_type}.compr"), "nnz.fraction"] = f"{nnzleaf_frac.mean():5.1f}×"
            dfraw.loc[(dname, model_type), "nnz.leafs"] = nnzleaf0.mean()
            dfraw.loc[(dname, f"{model_type}.compr"), "nnz.leafs"] = nnzleaf1.mean()

            time0 = np.array([j["clf_train_time"] for j in formodel.values()])
            time1 = np.array([j["compr_time"] for j in formodel.values()])
            df.loc[(dname, f"{model_type}.compr"), "time"] = f"{np.mean(time1):5.1f}s"
            dfraw.loc[(dname, model_type), "time"] = time0.mean()
            dfraw.loc[(dname, f"{model_type}.compr"), "time"] = time1.mean()

            #for k in ["bal.acc.train", "bal.acc.test", "nnz.leafs", "nnz.fraction"]:
            #    df.loc[(dname, " "), k] = " "


            #s = f"{dname} {model_type} {n}"
            #print(s)
            #print_metrics("TRA", mtrain0, mtrain1, abserr)
            #print_metrics("VAL", mvalid0, mvalid1, abserr)
            #print_metrics("TES", mtest0, mtest1, abserr)
            #print_nnzleafs("NNZ", nnzleaf0, nnzleaf1)

    print(dfraw.to_string())

    interesting = [
        "Albert",
        "California",
        "Spambase",
        "DefaultCredit",
        "BankMarketing",
        "Houses",
        "DryBean",
        "Abalone",
        "Yprop",
        "Vehicle",
        "Nomao",
    ]
    bad = [
        "RoadSafety",
        "Santander",
        "Banknote",
        "Bioresponse",
        "Adult",
        "Mercedes",
        "BreastCancer",
        "EyeMovements",
    ]
    dnames = []
    for n in organized.keys():
        n = n[0:15]
        for ni in interesting:
            if ni in n:
                n = f"\\textcolor{{olive}}{{{n}}}"
        for ni in bad:
            if ni in n:
                n = f"\\textcolor{{red}}{{{n}}}"
        dnames.append(n)

    df.index = pd.MultiIndex.from_product([dnames, ["XGB", "XGB.Compr", "DT", "DT.Cmpr"]])
    df.columns = ["n", "b.acc.train", "b.acc.test", "b.acc.test $\\Delta$", "nnz.leafs", "nnz.factor", "time"]

    df = df.drop(columns=["n"])

    print(df.to_string())

    #print(df.iloc[0:4])

    with open("tex/table.tex", "w") as fh:
        for k in range(0, len(df.index)//4, 14):
            #print(df.index[k*4:(k+14)*4])
            fh.write(df.iloc[k*4:(k+14)*4].to_latex(column_format="llrrlrrrr"))
            fh.write("\n\n")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    #ax.set_xscale("log")
    ax.hist(np.log10(nnz_fractions), bins=20)
    xticks = ax.get_xticks()
    ax.set_xticklabels([10**x for x in xticks])



    plt.show()
    
