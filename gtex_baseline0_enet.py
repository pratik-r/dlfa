# ENet for GTEx whole blood data with baseline functional annotations 

import os
import shutil
import numpy as np
import pandas as pd
from glmnet import ElasticNet
import sys
import lbfgs
from scipy.stats import spearmanr
from functools import partial
from loky import get_reusable_executor
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from read_gtex_baseline0_id import Loader

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

def get_nu(gamma, Amat):
    return 0.5*(1 + np.exp(-(Amat * gamma[None]).sum(axis=1)))

def fit_enet(X, y, Amat, gamma, alpha_, lambda_, random_state):
    nu = get_nu(gamma, Amat)
    model = ElasticNet(alpha = alpha_, lambda_path = [lambda_], n_splits = 0, random_state = random_state, standardize = False)
    model.fit(X / nu[np.newaxis,:], y)
    beta0 = model.intercept_path_[0]
    beta = model.coef_path_[:,0] / nu
    mse = ((beta0 + X @ beta - y)**2).sum() / (2*len(y))
    return beta0, beta, mse

def fit_enet_id(gene_id, gamma, alpha_, df_lambda_, loader, random_state):
    X_all, y_all, df_annot = loader.load_data(gene_id, load_annot = True)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = random_state)
    Amat = df_annot.values
    lambda_ = df_lambda_.loc[gene_id]
    try:
        beta0, beta, mse = fit_enet(X_train, y_train, Amat, gamma, alpha_, lambda_, random_state = random_state)
    except Exception as e:
        print(f"{gene_id}: ({type(e).__name__}) {e}")
        return
    y_pred_test = beta0 + X_test @ beta
    return pd.Series(
        {
            'pearson': np.corrcoef(y_pred_test, y_test)[0,1],
            'spearman': spearmanr(y_pred_test, y_test)[0],
            'Amat': Amat,
            'beta': beta,
            'num_features': np.sum(beta != 0),
            'lambda_': lambda_,
            'MSE': mse,
        }, name = gene_id)

def fit_enet_all(gamma, df_lambda_, loader, random_state):
    fit_enet_id_partial = partial(fit_enet_id, gamma = gamma, alpha_ = alpha_, df_lambda_ = df_lambda_, loader = loader, random_state = random_state)
    executor = get_reusable_executor()
    return pd.DataFrame(executor.map(fit_enet_id_partial, df_lambda_.index))

def loss_fn(gamma, eta, alpha_, df_res):
    df_res['nu'] = df_res.Amat.apply(lambda Amat: get_nu(gamma = gamma, Amat = Amat))
    loss = df_res.apply(lambda x: x.lambda_ * np.sum(x.nu * ((1-alpha_)*0.5*x.beta**2 + alpha_ * np.abs(x.beta))), axis = 1).mean() + eta * np.sum(np.abs(gamma))
    df_res.drop(columns = 'nu', inplace = True)
    return loss

def priler_loss(gamma, eta, alpha_, df_res):
    return df_res.MSE.mean() + loss_fn(gamma, eta, alpha_, df_res)

def objective(gamma, grad, alpha_, df_res):
    df_res['tmp'] = df_res.Amat.apply(lambda Amat: np.exp(-(Amat * gamma[None]).sum(axis=1))[:,np.newaxis])
    df_res['enet_penalty'] = df_res.apply(
        lambda x: (x.lambda_ * ((1-alpha_)*0.5 * x.beta**2 + alpha_ * np.abs(x.beta)))[:,np.newaxis],
        axis=1,
    )
    grad[:] = df_res.apply(
        lambda x: -0.5 * (x.Amat * x.tmp * x.enet_penalty).sum(axis = 0),
        axis=1
    ).mean()
    df_res.drop(columns = ['tmp','enet_penalty'], inplace = True)
    df_res['nu'] = df_res.Amat.apply(lambda Amat: get_nu(gamma = gamma, Amat = Amat))
    loss = df_res.apply(lambda x: x.lambda_ * np.sum(x.nu * ((1-alpha_)*0.5*x.beta**2 + alpha_ * np.abs(x.beta))), axis = 1).mean()
    df_res.drop(columns = 'nu', inplace = True)
    return loss

def inner_loop(gamma, eta, alpha_, df_res):
    f = partial(objective, alpha_ = alpha_, df_res = df_res)
    return lbfgs.fmin_lbfgs(f=f, x0=gamma, orthantwise_c = eta)

def save_df(df, filename):
    df1 = df[['pearson', 'spearman', 'num_features']].copy()
    df1.to_csv(os.path.join(save_dir, filename))

def save_losses(losses):
    plt.plot(losses)
    plt.savefig(os.path.join(save_dir, "losses.png"), bbox_inches = "tight")
    plt.close()
    pd.Series(losses).to_csv(os.path.join(save_dir, "losses.csv"), index=False)

def priler(df_lambda_, K, alpha_, eta, random_state, maxiter = 20, rel_tol = 1e-4):
    gamma = np.zeros(K)
    pd.Series(gamma).to_csv(os.path.join(save_dir, "gamma0.csv"), index=False)
    df_res = fit_enet_all(gamma, df_lambda_, loader, random_state = SEED)
    save_df(df_res, "res0.csv")
    loss_old = priler_loss(gamma, eta, alpha_, df_res)
    losses = [loss_old]
    for i in tqdm(range(maxiter)):
        gamma = inner_loop(gamma, eta, alpha_, df_res)
        pd.Series(gamma).to_csv(os.path.join(save_dir, f"gamma{i+1}.csv"), index=False)
        df_res = fit_enet_all(gamma, df_lambda_, loader, random_state = SEED)
        save_df(df_res, f"res{i+1}.csv")
        loss_new = priler_loss(gamma, eta, alpha_, df_res)
        losses.append(loss_new)
        save_losses(losses)
        if abs(loss_new - loss_old) / max([loss_old, loss_new]) < rel_tol:
            break
        loss_old = loss_new
    return df_res, gamma, losses

if __name__ == "__main__":
    SEED = 42
    chr_groups = [
        {'chr1', 'chr21'},
        {'chr2', 'chr3'},
        {'chr4', 'chr5', 'chr6'},
        {'chr7', 'chr8', 'chr9'},
        {'chr11', 'chr12', 'chr13'},
        {'chr14', 'chr15', 'chr16'},
        {'chr10', 'chr17', 'chr18'},
        {'chr19', 'chr20', 'chr22'},    
    ]

    eta = 5e-3
    alpha_ = 0.5
    maxiter = 50
    rel_tol = 1e-4
    
    ix = int(sys.argv[1])
    chr_group = chr_groups[ix]
    
    home_dir = # home directory 
    save_dir = os.path.join(home_dir, f"gtex_baseline0/{eta}/{ix}")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors = True)
    os.makedirs(save_dir)
    
    loader = Loader()
    df_hsq = loader.load_hsq()
    
    enet_path = os.path.join(home_dir, "vanilla_enet/df_enet.csv") # saved results from vanilla_enet
    df_enet = pd.read_csv(enet_path, index_col = 0)
    df_enet['chr'] = df_hsq.loc[df_enet.index, 'chr']
    
    df_genes = df_enet.loc[df_enet.chr.isin(chr_group)]
    df_lambda_ = df_genes.lambda_
    
    gene_id = df_lambda_.index[0]
    X_all, y_all, df_annot = loader.load_data(gene_id, load_annot = True)
    K = df_annot.shape[1]
    
    df_res, gamma, losses = priler(df_lambda_, K, alpha_, eta, random_state = SEED, maxiter = maxiter, rel_tol = rel_tol)