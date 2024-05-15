import os
import json
import click
import veritas
import numpy as np
import time
import warnings
import prada
from sklearn.metrics import zero_one_loss
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn


import util
import tree_compress


@click.group()
def cli():
    pass

@cli.command("list")
@click.option("--cmd", type=click.Choice(["train", "compress"]), default="train")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--penalty", type=click.Choice(["l1", "l2"]),
              default="l2")
@click.option("--seed", default=util.SEED)
def print_configs(cmd, linclf_type, penalty, seed):
    for dname in util.DNAMES:
        d = prada.get_dataset(dname, seed=seed, silent=True)

        for model_type in ["xgb"]:#, "dt"]:
            folds = [i for i in range(util.NFOLDS)]

            grid = d.paramgrid(fold=folds)

            for cli_param in grid:
                print("python can_experiment.py leaf_refine",
                      dname,
                      "--model_type", model_type,
                      "--linclf_type", linclf_type,
                      "--penalty", penalty,
                      "--fold", cli_param["fold"],
                      "--seed", seed,
                      "--silent")


@cli.command("leaf_refine")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--penalty", type=click.Choice(["l1", "l2", "lrl1"]),
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
        if penalty == "lrl1":
            refiner = LRPlusL1Refiner()
        elif penalty == "l2":
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
            

        sparse_train_x = transform_data_sparse(at_orig,dtrain.X) 
        sparse_valid_x = transform_data_sparse(at_orig,dvalid.X)


            
        if penalty == "lrl1":

            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.925,0.955,0.975,1]
            best_score = float('inf')
            for alpha in  alpha_list:
                refiner.set_params(at_orig, alpha)
                refiner.refine(50, sparse_train_x, dtrain.y)
                preds = refiner(torch.from_numpy(sparse_valid_x.todense())).detach().numpy()
                preds[preds > 0] = 1
                preds[preds <= 0] = -1
                score = zero_one_loss(dvalid.y,preds )
                if score < best_score:
                    best_score = score
                    best_alpha = alpha
            refiner.set_params(at_orig, best_alpha)
            refiner.refine(50, sparse_train_x, dtrain.y)
            
            at_refined = at_orig.copy()
            at_refined = set_new_addtree(refiner.tree_weights.detach().numpy(), [ w.detach().numpy() for w in refiner.leaf_weights], refiner.base_score.detach().numpy(), at_refined)
            

        else:
            alpha_list = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
            best_score = float('inf')
            for alpha in  alpha_list:
                refiner.set_params(C = 1/alpha)
                refiner.fit(sparse_train_x,dtrain.y)
                score = zero_one_loss(dvalid.y, refiner(sparse_valid_x))
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
            "best_alpha": best_alpha,

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
        "cmd": f"lr_{penalty}",
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
        "penalty": penalty,

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
    at.set_base_score(0, base_score)
    offset = 0
    for tree_index, t in enumerate(at):
        for count , leaf_id in enumerate(t.get_leaf_ids()):
            t.set_leaf_value(leaf_id, 0, new_leaf_vals[offset + count])
        offset += count + 1
    return at

def set_new_addtree(tree_weights, leaf_weights, base_score, at):
    new_at = veritas.AddTree(1, veritas.AddTreeType.REGR)
    new_at.set_base_score(0,base_score)
    for tree_index, t in enumerate(at):
        if tree_weights[tree_index] != 0:
            for count , leaf_id in enumerate(t.get_leaf_ids()):
                t.set_leaf_value(leaf_id, 0, leaf_weights[tree_index][count] * tree_weights[tree_index])
            new_at.add_tree(t)
    return new_at



def individual_contribution(i, ensemble_proba, target):
    """
    Compute the individual contributions of each classifier wrt. the entire ensemble. Return the negative contribution due to the minimization.

    Reference:
        Lu, Z., Wu, X., Zhu, X., & Bongard, J. (2010). Ensemble pruning via individual contribution ordering. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 871–880. https://doi.org/10.1145/1835804.1835914
    """
    iproba = ensemble_proba[i,:,:]
    n = iproba.shape[0]

    predictions = iproba.argmax(axis=1)
    V = np.zeros(ensemble_proba.shape)
    idx = ensemble_proba.argmax(axis=2)
    V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
    V = V.sum(axis=0)

    IC = 0
    #V = all_proba.argmax(axis=2)
    #predictions = iproba.argmax(axis=1)
    #V = all_proba.sum(axis=0)#.argmax(axis=1)

    for j in range(n):
        if (predictions[j] == target[j]):
            
            # case 1 (minority group)
            # label with majority votes on datapoint  = np.argmax(V[j, :]) 
            if(predictions[j] != np.argmax(V[j,:])):
                IC = IC + (2*(np.max(V[j,:])) - V[j, predictions[j]])
                
            else: # case 2 (majority group)
                # calculate second largest nr of votes on datapoint i
                sortedArray = np.sort(np.copy(V[j,:]))
                IC = IC + (sortedArray[-2])
                
        else:
            # case 3 (wrong prediction)
            IC = IC + (V[j, target[j]]  -  V[j, predictions[j]] - np.max(V[j,:]) )
    return - 1.0 * IC



class LRPlusL1Refiner(nn.Module):
    def __init__(self):
        super(LRPlusL1Refiner, self).__init__()
        self.at = None
        self.leaf_weights = None
        self.tree_weights = None
        self.base_score = None
        self.alpha = None
        #self.sm = nn.Softmax(dim=1)

    def set_params(self,at,alpha):
        self.at = at
        torch_leafs = []
        torch_trees = []
        for tree_index, t in enumerate(at):
            leaf_vals = []
            for l_id in t.get_leaf_ids():
                leaf_vals.append(t.get_leaf_value(l_id,0))
            leaf_vals = np.array(leaf_vals)
            torch_leafs.append(nn.Parameter(torch.from_numpy(leaf_vals)))
            torch_trees.append(1)
        self.leaf_weights = nn.ParameterList(torch_leafs)
        self.tree_weights= nn.Parameter(torch.Tensor(torch_trees))
        self.base_score = nn.Parameter(torch.tensor(at.get_base_score(0)))
        self.alpha = alpha

    def forward(self, x):
        offset = 0
        y = 0
        for i, p in enumerate(self.leaf_weights):
            #y += torch.sparse.mm(x[:,offset:len(p)], p) * self.tree_weights[i]
            y += torch.matmul(x[:,offset:len(p)+offset],p) * self.tree_weights[i]
            offset += len(p)
        y += self.base_score
        return y
    
    def refine(self, epochs, X, Y, batch_size = 128):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            mini_batches = create_mini_batches(X,Y, batch_size , True) 
            for x,y in mini_batches:
                optimizer.zero_grad()
                loss = criterion(self(x),y)
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    step_size = optimizer.param_groups[0]['lr']
                    proxed = psgd(self.tree_weights.numpy(),self.alpha,step_size)
                    self.tree_weights.copy_(torch.from_numpy(proxed))

def psgd(w, alpha, step_size):
    sign = np.sign(w)
    tmp_w = np.abs(w) - alpha*step_size
    tmp_w = sign*np.maximum(tmp_w,0)
    return tmp_w


def create_mini_batches(inputs, targets, batch_size, shuffle=False):
    """ Create an mini-batch like iterator for the given inputs / target / data. Shamelessly copied from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    
    Parameters
    ----------
    inputs : array-like vector or matrix 
        The inputs to be iterated in mini batches
    targets : array-like vector or matrix 
        The targets to be iterated in mini batches
    batch_size : int
        The mini batch size
    shuffle : bool, default False
        If True shuffle the batches 
    """
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    start_idx = 0
    while start_idx < len(indices):
        if start_idx + batch_size > len(indices) - 1:
            excerpt = indices[start_idx:]
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
        
        start_idx += batch_size

        #yield torch.sparse_coo_tensor(inputs[excerpt].nonzero(), inputs[excerpt].data, inputs[excerpt].shape), targets.iloc[excerpt]
        yield torch.from_numpy(inputs[excerpt].todense()), torch.tensor(targets.iloc[excerpt].values,dtype=torch.float64)


    


if __name__ == "__main__":
    cli()
