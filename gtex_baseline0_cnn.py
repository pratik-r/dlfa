# CNN for GTEx whole blood data with baseline functional annotations

import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from collections import ChainMap
from functools import partial
import lbfgs
import random

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

from read_gtex_baseline0_id import Loader
from cnn_fa import CNN

def get_nu(gamma, Amat):
    return 0.5*(1 + np.exp(-(Amat * gamma[None]).sum(axis=1)))

def make_loader(X, y, batch_size, random_state):
    Xt = torch.tensor(X.values, dtype = torch.float)
    yt = torch.tensor(y.values, dtype = torch.float)
    dataset = TensorDataset(Xt, yt)
    torch.manual_seed(random_state)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle = True)
    return data_loader

def compute_metrics(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X.values)).squeeze()
    return {
        'mse': mean_squared_error(y_pred, y),
        'pearson': np.corrcoef(y_pred, y)[0,1],
        'spearman': spearmanr(y_pred, y)[0],
    }

def compute_MSE(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X.values)).squeeze()
    return mean_squared_error(y_pred, y)

def fit_cnn(X_train, y_train, nu, hparams, random_state, init_params):
    train_loader = make_loader(X_train, y_train, batch_size = 32, random_state = random_state)
    d_in = X_train.shape[1]
    if nu is None:
        nu = torch.ones(d_in)
    hparams["nu"] = nu
    model = CNN(d_in = d_in, **hparams, random_state = random_state)
    model.load_state_dict(init_params, strict=False)
    trainer = pl.Trainer(log_every_n_steps = 1, max_epochs = hparams["max_epochs"], logger = False,
                         enable_checkpointing=False, enable_progress_bar = False)
    trainer.fit(model, train_loader)
    return model

def fit_cnn_id(gene_id, gamma, hparams, loader, random_state, init_dir):
    X_all, y_all, df_annot = loader.load_data(gene_id, load_annot = True)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = random_state)
    Amat = df_annot.values
    nu = torch.from_numpy(get_nu(gamma, Amat).astype(float))
    try:
        init_params = torch.load(os.path.join(init_dir, f"{gene_id}_init.pt"))
        model = fit_cnn(X_train, y_train, nu, hparams, random_state, init_params)
        metrics = compute_metrics(model, X_test, y_test)
        return pd.Series(
            ChainMap(*[metrics, 
                       {
                           'Amat': Amat,
                           'w': model.input_scaler.weight.data.numpy(),
                           'MSE': compute_MSE(model, X_train, y_train),
                       }]),
            name = gene_id,
        )
    except Exception as e:
        print(f"{gene_id}: ({type(e).__name__}) {e}")
        return
    
def fit_cnn_all(gamma, gene_ids, hparams, loader, random_state, init_dir):
    res = [fit_cnn_id(gene_id, gamma, hparams, loader, random_state, init_dir) for gene_id in gene_ids]
    return pd.DataFrame([x for x in res if x is not None])

def loss_fn(gamma, eta, hparams, df_res):
    df_res['nu'] = df_res.Amat.apply(lambda Amat: get_nu(gamma = gamma, Amat = Amat))
    lambda_ = hparams['lambda_reg']
    alpha_ = hparams['alpha_reg']
    loss = df_res.apply(lambda x: lambda_ * np.sum(x.nu * ((1-alpha_)*0.5*x.w**2 + alpha_ * np.abs(x.w))), axis = 1).mean() + eta * np.sum(np.abs(gamma))
    df_res.drop(columns = 'nu', inplace = True)
    return loss

def priler_loss(gamma, eta, hparams, df_res):
    return df_res.MSE.mean() + loss_fn(gamma, eta, hparams, df_res)

def objective(gamma, grad, hparams, df_res):
    lambda_ = hparams['lambda_reg']
    alpha_ = hparams['alpha_reg']
    df_res['tmp'] = df_res.Amat.apply(lambda Amat: np.exp(-(Amat * gamma[None]).sum(axis=1))[:,np.newaxis])
    df_res['weight_penalty'] = df_res.apply(
        lambda x: (lambda_ * ((1-alpha_)*0.5 * x.w**2 + alpha_ * np.abs(x.w)))[:,np.newaxis],
        axis=1,
    )
    grad[:] = df_res.apply(
        lambda x: -0.5 * (x.Amat * x.tmp * x.weight_penalty).sum(axis = 0),
        axis=1,
    ).mean()
    df_res.drop(columns = ['tmp','weight_penalty'], inplace = True)
    df_res['nu'] = df_res.Amat.apply(lambda Amat: get_nu(gamma = gamma, Amat = Amat))
    loss = df_res.apply(lambda x: lambda_ * np.sum(x.nu * ((1-alpha_)*0.5*x.w**2 + alpha_ * np.abs(x.w))), axis = 1).mean()
    df_res.drop(columns = 'nu', inplace = True)
    return loss

def inner_loop(gamma, eta, hparams, df_res):
    f = partial(objective, hparams = hparams, df_res = df_res)
    return lbfgs.fmin_lbfgs(f=f, x0=gamma, orthantwise_c = eta)

def save_df(df, filename):
    df1 = df[['pearson', 'spearman', 'w']].copy()
    df1['w'] = df1['w'].apply(lambda x: x.tolist())
    df1.to_csv(os.path.join(save_dir, filename))

def save_losses(losses):
    plt.plot(losses)
    plt.savefig(os.path.join(save_dir, "losses.png"), bbox_inches = "tight")
    plt.close()
    pd.Series(losses).to_csv(os.path.join(save_dir, "losses.csv"), index=False)
    
def priler(gene_ids, K, hparams, eta, random_state, init_dir, maxiter = 20, rel_tol = 1e-4):
    gamma = np.zeros(K)
    pd.Series(gamma).to_csv(os.path.join(save_dir, f"gamma0.csv"), index=False)
    df_res = fit_cnn_all(gamma, gene_ids, hparams, loader, random_state, init_dir)
    save_df(df_res, f"res0.csv")
    loss_old = priler_loss(gamma, eta, hparams, df_res)
    losses = [loss_old]
    for i in tqdm(range(maxiter)):
        gamma = inner_loop(gamma, eta, hparams, df_res)
        pd.Series(gamma).to_csv(os.path.join(save_dir, f"gamma{i+1}.csv"), index=False)
        df_res = fit_cnn_all(gamma, df_res.index.tolist(), hparams, loader, random_state, init_dir)
        save_df(df_res, f"res{i+1}.csv")
        loss_new = priler_loss(gamma, eta, hparams, df_res)
        losses.append(loss_new)
        save_losses(losses)
        if abs(loss_new - loss_old) / max([loss_old, loss_new]) < rel_tol:
            break
        loss_old = loss_new
    return df_res[['pearson','spearman', 'w']], gamma, losses

if __name__ == "__main__":
    eta = 1e-3
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

    ix = int(sys.argv[1])
    chr_group = chr_groups[ix]
    chr_group_dir = ix
    save_dir = # save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    init_dir = # CNN initial weights for each gene

    loader = Loader()
    df_hsq = loader.load_hsq()
    df_pearson = # correlations for CNN without FAs
    df_genes = df_hsq.loc[df_pearson.index]

    gene_id = df_genes.index[0]
    X_all, y_all, df_annot = loader.load_data(gene_id, load_annot = True)
    K = df_annot.shape[1]

    hparams = {
        'lr': 2e-3,
        'max_epochs': 100,
        'stem_features': 2,
        'depths': [1,1,1],
        'widths': [2,4,8],
        'expansion': 1,
        'weight_decay': 0.,
        'lambda_reg': 5e-2,
        'alpha_reg': 0.7,
        'drop_path': 0.8,
    }
    
    gene_ids = df_genes.index[df_genes['chr'].isin(chr_group)].tolist()
    print(f"num genes: {len(gene_ids)}")
    
    df_res, gamma, losses = priler(gene_ids, K, hparams, eta, SEED, init_dir, maxiter = 20, rel_tol = 1e-4)