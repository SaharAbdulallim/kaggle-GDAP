import glob
import os

import cv2
import numpy as np
import tifffile

from src.config import CFG


def load_sample(fname: str, hs_dir: str, ms_dir: str, rgb_dir: str):
    hs = tifffile.imread(os.path.join(hs_dir, fname + ".tif")).astype(np.float32)
    if hs.ndim == 2:
        return None
    if hs.shape[-1] > 125:
        hs = hs[:, :, :125]

    ms_path = os.path.join(ms_dir, fname + ".tif")
    ms = tifffile.imread(ms_path).astype(np.float32) if os.path.exists(ms_path) else None

    rgb_path = os.path.join(rgb_dir, fname + ".png")
    rgb = cv2.imread(rgb_path).astype(np.float32) if os.path.exists(rgb_path) else None

    return {"name": fname, "hs": hs, "ms": ms, "rgb": rgb}


def load_train(cfg: CFG):
    hs_dir = cfg.train_path("HS")
    ms_dir = cfg.train_path("MS")
    rgb_dir = cfg.train_path("RGB")
    cmap = cfg.class_map
    samples, labels = [], []

    for f in sorted(glob.glob(os.path.join(hs_dir, "*.tif*"))):
        fname = os.path.splitext(os.path.basename(f))[0]
        cls = fname.split("_hyper_")[0]
        if cls not in cmap or fname in cfg.BLANK_SAMPLES:
            continue
        s = load_sample(fname, hs_dir, ms_dir, rgb_dir)
        if s is None:
            continue
        s["cls"] = cls
        s["label"] = cmap[cls]
        samples.append(s)
        labels.append(cmap[cls])

    return samples, np.array(labels)


def load_test(cfg: CFG):
    hs_dir = cfg.test_path("HS")
    ms_dir = cfg.test_path("MS")
    rgb_dir = cfg.test_path("RGB")
    samples = []

    for f in sorted(glob.glob(os.path.join(hs_dir, "*.tif*"))):
        fname = os.path.splitext(os.path.basename(f))[0]
        s = load_sample(fname, hs_dir, ms_dir, rgb_dir)
        if s is None:
            continue
        samples.append(s)

    return samples
