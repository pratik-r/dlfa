# ENet for GTEx whole blood data

import os
import numpy as np
import pandas as pd
from glmnet import ElasticNet
import sys
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from tqdm import tqdm

from read_gtex_brain_baseline0_ix import Loader

def fit_enet(X, y, alpha_, random_state):
    model_enet = ElasticNet(alpha = alpha_, n_splits = 3, random_state = random_state, standardize = False, cut_point = 0)
    model_enet.fit(X, y)
    beta0 = model_enet.intercept_
    beta = model_enet.coef_
    lambda_ = model_enet.lambda_best_.item()
    return beta0, beta, lambda_

def fit_enet_ix(ix, loader, alpha_, random_state):
    gene_id, X_all, y_all, df_annot = loader.load_data(ix, load_annot = False)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.2, random_state = random_state)
    beta0, beta, lambda_ = fit_enet(X_train, y_train, alpha_, random_state = random_state)
    y_pred_test = beta0 + X_test @ beta
    return pd.Series(
        {
            'pearson': np.corrcoef(y_pred_test, y_test)[0,1],
            'spearman': spearmanr(y_pred_test, y_test)[0],
            'num_features': np.sum(beta != 0),
            'P': len(beta),
            'lambda_': lambda_,
        }, name = gene_id)

if __name__ == "__main__":
    SEED = 42
    alpha_ = 0.5
    save_dir = # save directory
    
    loader = Loader()
    df_hsq = loader.load_hsq()
    num_array_splits = 2000
    ixs_splits = np.array_split(np.arange(df_hsq.shape[0]), num_array_splits)
    
    ixs = ixs_splits[int(sys.argv[1])]
    for ix in tqdm(ixs):
        try:
            res = fit_enet_ix(ix, loader, alpha_, random_state = SEED)
            res.to_csv(os.path.join(save_dir, f"{res.name}.csv"))
        except:
            continue