import torch
from tqdm import tqdm

from src.utils import (
    make_df,
    read_hs,
    read_ms,
    read_rgb,
    resize_tensor,
    stratified_holdout,
)


def calculate_stats(cfg, verbose=True):
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
    
    if verbose:
        print("Calculated Statistics:")

        if 'rgb_mean' in stats:
            print(f"\nRGB_MEAN: tuple = {stats['rgb_mean']}")
            print(f"RGB_STD: tuple = {stats['rgb_std']}")
        if 'ms_mean' in stats:
            print(f"\nMS_MEAN: tuple = {stats['ms_mean']}")
            print(f"MS_STD: tuple = {stats['ms_std']}")
        if 'hs_mean' in stats:
            print(f"\nHS_MEAN: tuple = {stats['hs_mean']}")
            print(f"HS_STD: tuple = {stats['hs_std']}")

        print("ImageNet default stats (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) recommended for RGB with pretrained weights.")
    
    return stats


if __name__ == "__main__":
    from src.config import CFG
    cfg = CFG()
    cfg.ROOT = "./data"
    cfg.TRAIN_DIR = "train"
    calculate_stats(cfg)
