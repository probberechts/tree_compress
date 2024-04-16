import glob
import json
import colorama
import numpy as np
import pandas as pd

LBRACE = "{"
RBRACE = "}"


def read_results(abserr, seed):
    jsons = []
    for f in glob.glob("results/*"):
        with open(f) as fh:
            for line in fh.readlines():
                if not line.startswith(LBRACE):
                    continue

                j = json.loads(line)
                if j["seed"] != seed:
                    continue
                if j["abserr"] != abserr:
                    continue

                jsons.append(j)


    organized = {}

    for j in jsons:
        dname = j["dname"]
        model_type = j["model_type"]
        fold = j["fold"]

        if dname not in organized:
            organized[dname] = {}
        fordata = organized.get(dname, {})
        if model_type not in fordata:
            fordata[model_type] = {}
        formodel = fordata.get(model_type, {})

        formodel[fold] = j

    return organized


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
    
