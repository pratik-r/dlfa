# class to load clean GTEx data and baseline functional annotations by gene ID

import os
import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin
from functools import reduce

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

gtex_path = # path to GTEx dir
home_path = # home path
gtex_exp_path = # path to normalized gene expression file
hsq_path = # path to list of 
gtex_plink_dir = # path to SNP data processed by Plink

class Loader:
    def __init__(self):
        pass
        
    def load_hsq(self, columns = ["chr", "HSQ"]):
        return pd.read_parquet(hsq_path, columns = columns)
    
    def load_data(self, gene_id, load_annot):
        df_hsq = self.load_hsq()
        chr_ = df_hsq.loc[gene_id, 'chr']
        
        gtex_plink_files = os.listdir(gtex_plink_dir)
        gtex_gene_files = {file.split('.')[-1]: file for file in gtex_plink_files if f'{gene_id}.' in file}
        gtex_gene_files = {k: os.path.join(gtex_plink_dir,v) for k,v in gtex_gene_files.items() if k in ['bed','bim','fam']}
        gtex_snps = read_plink1_bin(**gtex_gene_files, verbose=False)
        gtex_snps_df = gtex_snps.to_pandas()
        gtex_snps_df.rename(columns = dict(zip(gtex_snps.variant.to_numpy(), gtex_snps.snp.to_numpy())), inplace=True)
        gtex_lookup = pd.read_csv(os.path.join(gtex_plink_dir, f"lookup/{chr_}.csv"), index_col=0).squeeze()
        gtex_snps_df.rename(columns = gtex_lookup, inplace=True)
        gtex_rsids = set(gtex_snps_df.columns).difference({'.'})
        
        annot_path = os.path.join(home_path, f'annot_baseline0_scaled/baseline_{chr_}.csv') # path to functional annotations file
        annot_rsids_raw = pd.read_csv(annot_path, usecols = [0]).squeeze()
        annot_rsids_set = set(annot_rsids_raw)

        rsids = reduce(set.intersection, [gtex_rsids, annot_rsids_set])

        gtex_snps_df_clean = gtex_snps_df.loc[:,gtex_snps_df.columns.isin(rsids)]
        gtex_snps_df_clean = gtex_snps_df_clean.loc[:,gtex_snps_df_clean.columns.sort_values()]

        # get residual expression (after regressing against covariates)
        y_gtex_res_scaled = pd.read_parquet(gtex_exp_path, columns = [gene_id]).squeeze()
        X_gtex = gtex_snps_df_clean.loc[y_gtex_res_scaled.index]

        # annotations
        if load_annot:
            annot_rows = [0] + (1+np.where(annot_rsids_raw.isin(rsids))[0]).tolist()
            df_annot = pd.read_csv(annot_path, skiprows = lambda x: x not in annot_rows, header = 0).set_index('rs').sort_index()
        else:
            df_annot = None

        return X_gtex, y_gtex_res_scaled, df_annot        
        