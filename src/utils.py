import os
import random
import re
from typing import Dict, Optional

import cv2
import kornia.augmentation as K
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tifffile as tiff
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import CFG, ID2LBL, LBL2ID


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
    def __init__(self, df: pd.DataFrame, cfg: CFG, hs_ch: int, transforms=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.hs_ch = hs_ch
        self.transforms = transforms
        self.concat_mode = cfg.CONCAT_MODE
        
        self.rgb_ch = 3 if cfg.USE_RGB else 0
        self.ms_ch = 5 if cfg.USE_MS else 0
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
        
        if self.concat_mode:
            tensors = [m for m in [modalities["rgb"], modalities["ms"], modalities["hs"]] if m is not None]
            img = torch.cat(tensors, 0) if tensors else torch.zeros(self.total_ch, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE)
            
            if self.transforms:
                img = self.transforms(img.unsqueeze(0)).squeeze(0)
            
            if "label" in row:
                return img, torch.tensor(LBL2ID[row["label"]], dtype=torch.long)
            return img, row["base_id"]
        
        else:
            if modalities["rgb"] is None:
                modalities["rgb"] = torch.zeros(3, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE)
            if modalities["ms"] is None:
                modalities["ms"] = torch.zeros(5, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE)
            if modalities["hs"] is None:
                modalities["hs"] = torch.zeros(self.hs_ch, self.cfg.IMG_SIZE, self.cfg.IMG_SIZE)
            
            if self.transforms:
                stacked = torch.cat([modalities["rgb"], modalities["ms"], modalities["hs"]], 0)
                stacked = self.transforms(stacked.unsqueeze(0)).squeeze(0)
                modalities["rgb"] = stacked[:3]
                modalities["ms"] = stacked[3:8]
                modalities["hs"] = stacked[8:]
            
            if "label" in row:
                return modalities, torch.tensor(LBL2ID[row["label"]], dtype=torch.long)
            return modalities, row["base_id"]


class WheatDataModule(pl.LightningDataModule):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        
        self.train_transforms = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=90.0, p=0.5),
            data_keys=["image"]
        )
        
    def setup(self, stage=None):
        train_df = make_df(self.cfg.ROOT, self.cfg.TRAIN_DIR)
        self.test_df = make_df(self.cfg.ROOT, self.cfg.VAL_DIR)
        
        self.hs_ch = infer_hs_channels(train_df, self.cfg)
        self.n_ch = (3 if self.cfg.USE_RGB else 0) + (5 if self.cfg.USE_MS else 0) + (self.hs_ch if self.cfg.USE_HS else 0)
        
        df_tr, df_va = stratified_holdout(train_df, frac=0.1, seed=self.cfg.SEED)
        
        self.train_ds = WheatDataset(df_tr, self.cfg, self.hs_ch, transforms=self.train_transforms)
        self.val_ds = WheatDataset(df_va, self.cfg, self.hs_ch, transforms=None)
        self.test_ds = WheatDataset(self.test_df, self.cfg, self.hs_ch, transforms=None)
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=True,
                         num_workers=self.cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=False,
                         num_workers=self.cfg.NUM_WORKERS, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=False,
                         num_workers=self.cfg.NUM_WORKERS, pin_memory=True)