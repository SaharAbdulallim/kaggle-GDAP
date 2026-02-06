import os
import random
import re
from typing import Dict, Optional

import cv2
import joblib
import kornia.augmentation as K
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tifffile as tiff
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import CFG, LBL2ID


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_label(filename: str) -> Optional[str]:
    m = re.match(r"^(Health|Rust|Other)_", filename)
    return m.group(1) if m else None


def make_df(root: str, split: str) -> pd.DataFrame:
    split_dir = os.path.join(root, split)
    rgb_dir = os.path.join(split_dir, "RGB")
    ms_dir = os.path.join(split_dir, "MS")
    hs_dir = os.path.join(split_dir, "HS")
    
    data = {}
    
    if os.path.isdir(rgb_dir):
        for fn in os.listdir(rgb_dir):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_id = os.path.splitext(fn)[0]
                data.setdefault(base_id, {})['rgb'] = os.path.join(rgb_dir, fn)
    
    if os.path.isdir(ms_dir):
        for fn in os.listdir(ms_dir):
            if fn.lower().endswith(('.tif', '.tiff')):
                base_id = os.path.splitext(fn)[0]
                data.setdefault(base_id, {})['ms'] = os.path.join(ms_dir, fn)
    
    if os.path.isdir(hs_dir):
        for fn in os.listdir(hs_dir):
            if fn.lower().endswith(('.tif', '.tiff')):
                base_id = os.path.splitext(fn)[0]
                data.setdefault(base_id, {})['hs'] = os.path.join(hs_dir, fn)
    
    rows = []
    for base_id, paths in data.items():
        row = {'base_id': base_id, **paths}
        if split == 'train':
            label = parse_label(base_id)
            if label:
                row['label'] = label
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if split == 'train':
        df = df.dropna(subset=['label'])
    return df


def stratified_holdout(df: pd.DataFrame, frac: float = 0.1, seed: int = 42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_va = pd.concat([g.iloc[:max(1, int(len(g) * frac))] for _, g in df.groupby("label")])
    df_tr = df[~df["base_id"].isin(df_va["base_id"])].reset_index(drop=True)
    return df_tr.reset_index(drop=True), df_va.reset_index(drop=True)


def read_rgb(path: str) -> torch.Tensor:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def read_tiff(path: str) -> np.ndarray:
    arr = tiff.imread(path)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, 0.0).astype(np.float32)
    flat = x.reshape(-1, x.shape[2])
    mn, mx = flat.min(0), flat.max(0)
    denom = mx - mn
    denom[denom < 1e-6] = 1.0
    return np.clip((x - mn) / denom, 0, 1)


def read_ms(path: str) -> torch.Tensor:
    return torch.from_numpy(normalize_minmax(read_tiff(path))).permute(2, 0, 1)


def read_hs(path: str, drop_first: int, drop_last: int) -> torch.Tensor:
    arr = read_tiff(path)
    if arr.shape[2] > drop_first + drop_last + 1:
        arr = arr[:, :, drop_first:-drop_last if drop_last > 0 else arr.shape[2]]
    return torch.from_numpy(normalize_minmax(arr)).permute(2, 0, 1)


def resize_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)


def infer_hs_channels(df: pd.DataFrame, cfg: CFG) -> int:
    if 'hs' in df.columns:
        for p in df["hs"].dropna():
            if os.path.exists(p):
                return int(read_hs(p, cfg.HS_DROP_FIRST, cfg.HS_DROP_LAST).shape[0])
    return 101


class WheatDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: CFG, hs_ch: int, transforms=None, pca_model=None, pca_n_features=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.hs_ch = hs_ch
        self.transforms = transforms
        self.pca_model = pca_model
        self.pca_n_features = pca_n_features if pca_n_features is not None else hs_ch
        self.pca_n_features = pca_n_features if pca_n_features is not None else hs_ch
        
        self.rgb_ch = 3 if cfg.USE_RGB else 0
        self.ms_ch = 5 if cfg.USE_MS else 0
        
        if pca_model is not None and cfg.USE_HS:
            self.hs_ch_used = cfg.PCA_COMPONENTS
        else:
            self.hs_ch_used = hs_ch if cfg.USE_HS else 0
        
        self.total_ch = self.rgb_ch + self.ms_ch + self.hs_ch_used
    
    def __len__(self):
        return len(self.df)
    
    def _load_modalities(self, row) -> Dict[str, Optional[torch.Tensor]]:
        modalities = {}
        
        if self.cfg.USE_RGB and pd.notna(row.get("rgb")):
            x = read_rgb(row["rgb"])
            x = resize_tensor(x, self.cfg.IMG_SIZE)
            modalities["rgb"] = x
        else:
            modalities["rgb"] = None
        
        if self.cfg.USE_MS and pd.notna(row.get("ms")):
            x = read_ms(row["ms"])
            x = resize_tensor(x, self.cfg.IMG_SIZE)
            modalities["ms"] = x
        else:
            modalities["ms"] = None
        
        if self.cfg.USE_HS and pd.notna(row.get("hs")):
            x = read_hs(row["hs"], self.cfg.HS_DROP_FIRST, self.cfg.HS_DROP_LAST)
            
            if self.pca_model is not None:
                target_ch = self.pca_n_features
                if x.shape[0] < target_ch:
                    pad = torch.zeros(target_ch - x.shape[0], *x.shape[1:])
                    x = torch.cat([x, pad], 0)
                elif x.shape[0] > target_ch:
                    x = x[:target_ch]
                x = resize_tensor(x, self.cfg.IMG_SIZE)
                C, H, W = x.shape
                x_flat = x.permute(1, 2, 0).reshape(-1, C).numpy()
                x_pca = self.pca_model.transform(x_flat)
                x = torch.from_numpy(x_pca.reshape(H, W, self.cfg.PCA_COMPONENTS)).permute(2, 0, 1).float()
            else:
                if x.shape[0] < self.hs_ch:
                    pad = torch.zeros(self.hs_ch - x.shape[0], *x.shape[1:])
                    x = torch.cat([x, pad], 0)
                elif x.shape[0] > self.hs_ch:
                    x = x[:self.hs_ch]
                x = resize_tensor(x, self.cfg.IMG_SIZE)
            
            modalities["hs"] = x
        else:
            modalities["hs"] = None
        
        return modalities
    
    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        modalities = self._load_modalities(row)
        
        if modalities["rgb"] is None:
            modalities["rgb"] = torch.zeros(3, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE)
        if modalities["ms"] is None:
            modalities["ms"] = torch.zeros(5, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE)
        if modalities["hs"] is None:
            modalities["hs"] = torch.zeros(self.hs_ch_used, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE)
        
        if self.transforms:
            stacked = torch.cat([modalities["rgb"], modalities["ms"], modalities["hs"]], 0)
            stacked = self.transforms(stacked.unsqueeze(0)).squeeze(0)
            modalities["rgb"] = stacked[:3]
            modalities["ms"] = stacked[3:8]
            modalities["hs"] = stacked[8:8+self.hs_ch_used]
        
        if self.cfg.RGB_MEAN is not None and self.cfg.RGB_STD is not None:
            mean = torch.tensor(self.cfg.RGB_MEAN).view(3, 1, 1)
            std = torch.tensor(self.cfg.RGB_STD).view(3, 1, 1)
            modalities["rgb"] = (modalities["rgb"] - mean) / std
        
        if self.cfg.MS_MEAN is not None and self.cfg.MS_STD is not None:
            mean = torch.tensor(self.cfg.MS_MEAN).view(5, 1, 1)
            std = torch.tensor(self.cfg.MS_STD).view(5, 1, 1)
            modalities["ms"] = (modalities["ms"] - mean) / std
        
        if self.cfg.HS_MEAN is not None and self.cfg.HS_STD is not None:
            hs_mean = torch.tensor(self.cfg.HS_MEAN)[:self.hs_ch_used]
            hs_std = torch.tensor(self.cfg.HS_STD)[:self.hs_ch_used]
            mean = hs_mean.view(self.hs_ch_used, 1, 1)
            std = hs_std.view(self.hs_ch_used, 1, 1)
            modalities["hs"] = (modalities["hs"] - mean) / std
        
        if "label" in row:
            return modalities, torch.tensor(LBL2ID[row["label"]], dtype=torch.long)
        return modalities, row["base_id"]


class WheatDataModule(pl.LightningDataModule):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.train_transforms = self._get_train_transforms()
    
    def _get_train_transforms(self):
        augs = [
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=90.0, p=0.5),
            K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.3),
            K.RandomGaussianNoise(mean=0., std=0.03, p=0.2)
        ]
        return K.AugmentationSequential(*augs, data_keys=["image"])
        
    def setup(self, stage=None):
        train_df = make_df(self.cfg.ROOT, self.cfg.TRAIN_DIR)
        self.test_df = make_df(self.cfg.ROOT, self.cfg.VAL_DIR)
        
        self.hs_ch = infer_hs_channels(train_df, self.cfg)
        
        pca_model = None
        pca_n_features = None
        if self.cfg.PCA_COMPONENTS > 0 and self.cfg.USE_HS:
            if os.path.exists(self.cfg.PCA_PATH):
                pca_data = joblib.load(self.cfg.PCA_PATH)
                if isinstance(pca_data, dict):
                    pca_model = pca_data['model']
                    pca_n_features = pca_data['n_features']
                else:
                    pca_model = pca_data
                    pca_n_features = getattr(pca_model, 'n_features_expected', self.hs_ch)
                hs_ch_out = self.cfg.PCA_COMPONENTS
            else:
                print(f"Warning: PCA not found at {self.cfg.PCA_PATH}. Run stats.py first.")
                hs_ch_out = self.hs_ch
        else:
            hs_ch_out = self.hs_ch
        
        self.n_ch = (3 if self.cfg.USE_RGB else 0) + (5 if self.cfg.USE_MS else 0) + (hs_ch_out if self.cfg.USE_HS else 0)
        
        df_tr, df_va = stratified_holdout(train_df, frac=0.1, seed=self.cfg.SEED)
        
        self.train_ds = WheatDataset(df_tr, self.cfg, self.hs_ch, transforms=self.train_transforms, pca_model=pca_model, pca_n_features=pca_n_features)
        self.val_ds = WheatDataset(df_va, self.cfg, self.hs_ch, transforms=None, pca_model=pca_model, pca_n_features=pca_n_features)
        self.test_ds = WheatDataset(self.test_df, self.cfg, self.hs_ch, transforms=None, pca_model=pca_model, pca_n_features=pca_n_features)
        
        self.hs_ch = hs_ch_out
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=True,
                         num_workers=self.cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=False,
                         num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=False,
                         num_workers=self.cfg.NUM_WORKERS, pin_memory=True)