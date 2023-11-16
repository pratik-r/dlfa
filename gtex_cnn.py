# CNN for GTEx whole blood data

import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import ElasticNetCV
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import spearmanr

import pytorch_lightning as pl

from read_gtex_baseline0_ix import Loader
from cnn import CNN

def fit_enet(X_train, y_train, X_test, y_test, n_folds, random_state):
    model = ElasticNetCV(l1_ratio = 0.5, random_state = random_state, cv = n_folds)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    coefs = model.coef_
    return {
        'mse': MSE(y_test_pred, y_test),
        'pearson': np.corrcoef(y_test_pred, y_test)[0,1],
        'spearman': spearmanr(y_test_pred, y_test)[0],
    }

def compute_metrics(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X.values)).squeeze()
    return {
        'mse': MSE(y_pred, y),
        'pearson': np.corrcoef(y_pred, y)[0,1],
        'spearman': spearmanr(y_pred, y)[0],
    }

def compute_metrics_ensemble(models, X, y):
    n = len(models)
    y_pred = torch.zeros(X.shape[0])
    for model in models:
        with torch.no_grad():
            y_pred += model(torch.tensor(X.values)).squeeze()
    y_pred /= n
    return {
        'mse': MSE(y_pred, y),
        'pearson': np.corrcoef(y_pred, y)[0,1],
        'spearman': spearmanr(y_pred, y)[0],
    }

def make_loader(X, y, batch_size, random_state):
    Xt = torch.tensor(X.values, dtype = torch.float)
    yt = torch.tensor(y.values, dtype = torch.float)
    dataset = TensorDataset(Xt, yt)
    torch.manual_seed(random_state)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle = True)
    return data_loader

def fit_cnn(d_in, train_loader, X_val, y_val, hparams, random_state, save_init):
    model = CNN(d_in = d_in, **hparams, random_state = random_state, save_init = save_init)
    trainer = pl.Trainer(log_every_n_steps = 1, max_epochs = hparams["max_epochs"], logger = False,
                         enable_checkpointing=False, enable_progress_bar = False)
    trainer.fit(model, train_loader)
    out = pd.Series(compute_metrics(model, X_val, y_val))
    out['model'] = model
    return out

if __name__ == "__main__":
    save_dir = # save directory
    
    loader = Loader()
    df_hsq = loader.load_hsq()
    num_array_splits = 2000
    ixs_splits = np.array_split(np.arange(df_hsq.shape[0]), num_array_splits)

    ixs = ixs_splits[int(sys.argv[1])]
    SEED = 42
    numiter = 20
    n_folds = 5
    nbest = 5
    
    hparams = {
        'lr': 2e-3,
        'max_epochs': 100,
        'stem_features': 2,
        'depths': [1,1,1],
        'widths': [2,4,8],
        'expansion': 1,
        'weight_decay': 1e-5,
        'lambda_reg': 5e-2,
        'alpha_reg': 0.7,
        'drop_path': 0.8,
    }

    for ix in tqdm(ixs):
        try:
            gene_id, X_all, y_all, df_annot = loader.load_data(ix, load_annot=False)
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = SEED)
            d_in = X_train.shape[1]

            metrics_test_enet = pd.Series(fit_enet(X_train, y_train, X_test, y_test, n_folds, SEED))

            kf = KFold(n_splits = n_folds, shuffle = True, random_state = SEED)
            np.random.seed(SEED)
            seeds = np.array_split(np.random.choice(range(numiter * n_folds), size = numiter*n_folds, replace=False), n_folds)

            df_models_dict = dict()
            for fold, (train_ix, val_ix) in enumerate(kf.split(X_train)):
                X_train1, X_val = X_train.iloc[train_ix], X_train.iloc[val_ix]
                y_train1, y_val = y_train.iloc[train_ix], y_train.iloc[val_ix]
                train_loader = make_loader(X_train1, y_train1, batch_size = 32, random_state = SEED)
                df_models_dict[fold] = pd.concat(
                    [fit_cnn(d_in, train_loader, X_val, y_val, hparams, random_state = seeds[fold][i], save_init = True) for i in range(numiter)],
                    axis=1).T
            df_models = pd.concat(df_models_dict, axis = 0)

            df_models_nbest = df_models.groupby(level = 0, group_keys = False).apply(lambda df: df.sort_values('pearson', ascending=False).iloc[:nbest])
            df_models_best = df_models_nbest.groupby(level = 0, group_keys=False).apply(lambda df: df.iloc[0])
            model_best = df_models_best.loc[df_models_best.pearson.idxmax(), 'model']

            metrics_test_nbest = pd.Series(compute_metrics_ensemble(df_models_nbest.model.tolist(), X_test, y_test))
            metrics_test_best = pd.Series(compute_metrics_ensemble(df_models_best.model.tolist(), X_test, y_test))
            metrics_test_best1 = pd.Series(compute_metrics(model_best, X_test, y_test))

            df_res = pd.DataFrame.from_dict({
                'nbest': metrics_test_nbest,
                'best': metrics_test_best,
                'best1': metrics_test_best1,
                'enet': metrics_test_enet,
            }, orient='index')
            df_res.to_csv(os.path.join(save_dir, f"{gene_id}_res.csv"))

            torch.save(model_best.init_params, os.path.join(save_dir, f"{gene_id}_init.pt"))
        except:
            continue