import os

import joblib
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.utils import (
    make_df,
    read_hs,
    read_ms,
    read_rgb,
    resize_tensor,
    stratified_holdout,
)


def calculate_stats(cfg, verbose=True, fit_pca=False, pca_path="pca_hs.pkl"):
    train_df = make_df(cfg.ROOT, cfg.TRAIN_DIR)
    train_df, _ = stratified_holdout(train_df, frac=0.1, seed=cfg.SEED)
    
    if verbose:
        print(f"Calculating statistics from {len(train_df)} samples...")
    
    rgb_vals, ms_vals, hs_vals, hs_max_ch = [], [], [], 0
    
    for idx in tqdm(range(len(train_df)), desc="Loading", disable=not verbose):
        row = train_df.iloc[idx]
        
        if cfg.USE_RGB and 'rgb' in row and row['rgb']:
            rgb_vals.append(resize_tensor(read_rgb(row['rgb']), cfg.IMG_SIZE))
        
        if cfg.USE_MS and 'ms' in row and row['ms']:
            ms_vals.append(resize_tensor(read_ms(row['ms']), cfg.IMG_SIZE))
        
        if cfg.USE_HS and 'hs' in row and row['hs']:
            hs = resize_tensor(read_hs(row['hs'], cfg.HS_DROP_FIRST, cfg.HS_DROP_LAST), cfg.IMG_SIZE)
            hs_vals.append(hs)
            hs_max_ch = max(hs_max_ch, hs.shape[0])
    
    stats = {}
    
    if rgb_vals:
        rgb_tensor = torch.stack(rgb_vals)
        stats['rgb_mean'] = tuple(rgb_tensor.mean(dim=[0, 2, 3]).tolist())
        stats['rgb_std'] = tuple(rgb_tensor.std(dim=[0, 2, 3]).tolist())
    
    if ms_vals:
        ms_tensor = torch.stack(ms_vals)
        stats['ms_mean'] = tuple(ms_tensor.mean(dim=[0, 2, 3]).tolist())
        stats['ms_std'] = tuple(ms_tensor.std(dim=[0, 2, 3]).tolist())
    
    if hs_vals:
        if hs_max_ch > min(hs.shape[0] for hs in hs_vals):
            hs_padded = [torch.cat([hs, torch.zeros(hs_max_ch - hs.shape[0], *hs.shape[1:])], 0) if hs.shape[0] < hs_max_ch else hs for hs in hs_vals]
        else:
            hs_padded = hs_vals
        hs_tensor = torch.stack(hs_padded)
        stats['hs_mean'] = tuple(hs_tensor.mean(dim=[0, 2, 3]).tolist())
        stats['hs_std'] = tuple(hs_tensor.std(dim=[0, 2, 3]).tolist())
        stats['hs_channels'] = hs_max_ch
        
        if fit_pca and hasattr(cfg, 'PCA_COMPONENTS') and cfg.PCA_COMPONENTS > 0:
            if verbose:
                print(f"\nFitting PCA: {hs_max_ch} channels → {cfg.PCA_COMPONENTS} components...")
            
            hs_flat = hs_tensor.permute(0, 2, 3, 1).reshape(-1, hs_max_ch).numpy()
            pca = PCA(n_components=cfg.PCA_COMPONENTS)
            pca.n_features_expected = hs_max_ch
            pca.fit(hs_flat)
            
            explained_var = pca.explained_variance_ratio_.sum()
            if verbose:
                print(f"Explained variance: {explained_var:.2%}")
            
            hs_pca = pca.transform(hs_flat).reshape(len(hs_tensor), cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.PCA_COMPONENTS)
            hs_pca_tensor = torch.from_numpy(hs_pca).permute(0, 3, 1, 2).float()
            
            os.makedirs(os.path.dirname(pca_path) if os.path.dirname(pca_path) else '.', exist_ok=True)
            pca_data = {
                'model': pca,
                'n_features': hs_max_ch,
                'ms_mean': stats.get('ms_mean'),
                'ms_std': stats.get('ms_std'),
                'hs_pca_mean': tuple(hs_pca_tensor.mean(dim=[0, 2, 3]).tolist()),
                'hs_pca_std': tuple(hs_pca_tensor.std(dim=[0, 2, 3]).tolist()),
                'pca_explained_variance': float(explained_var)
            }
            joblib.dump(pca_data, pca_path)
            if verbose:
                print(f"PCA saved: {pca_path} | Variance: {explained_var:.1%}")
            
            stats['hs_pca_mean'] = pca_data['hs_pca_mean']
            stats['hs_pca_std'] = pca_data['hs_pca_std']
            stats['pca_explained_variance'] = pca_data['pca_explained_variance']
    
    return stats


if __name__ == "__main__":
    from src.config import CFG
    cfg = CFG()
    cfg.ROOT = "./data"
    cfg.TRAIN_DIR = "train"
    cfg.PCA_COMPONENTS = 20
    calculate_stats(cfg, fit_pca=True, pca_path="./outputs/pca_hs.pkl")
