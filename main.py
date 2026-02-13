import argparse
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CFG
from src.data import load_test, load_train
from src.features import extract_batch
from src.optimize import (
    evaluate,
    predict,
    run_optimization,
    train_final,
)

parser = argparse.ArgumentParser()
parser.add_argument("--run-optuna", action="store_true", help="Run Optuna optimization")
parser.add_argument("--trials", type=int, default=40, help="Number of Optuna trials")
parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
parser.add_argument("--features", type=int, default=120, help="Number of top features")
args = parser.parse_args()

cfg = CFG()
cfg.OPTUNA_TRIALS = args.trials
cfg.CV_FOLDS = args.folds
cfg.N_TOP_FEATURES = args.features

print("Loading data...")
samples, labels = load_train(cfg)
test_samples = load_test(cfg)
counts = dict(zip(*np.unique(labels, return_counts=True)))
print(f"Train: {len(samples)} samples | {counts}")
print(f"Test:  {len(test_samples)} samples")

print("\nExtracting features...")
X_all = extract_batch(samples)
X_test_all = extract_batch(test_samples)
print(f"Train features: {X_all.shape}")
print(f"Test features:  {X_test_all.shape}")

if args.run_optuna:
    print("\nOptuna HPO (per-fold feature selection)...")
    result = run_optimization(X_all, labels, cfg)
    cfg.LGB_PARAMS = result["params"]
    cfg.VAR_THRESHOLD = result["var_threshold"]
    cfg.N_TOP_FEATURES = result["n_features"]
    print(f"Best F1: {result['best_f1']:.4f}  |  Gap: {result['best_gap']:.4f}")
    print(f"Params: {cfg.LGB_PARAMS}")
    print(f"Var threshold: {cfg.VAR_THRESHOLD:.2e}")
    print(f"N features: {cfg.N_TOP_FEATURES}")
else:
    print("\nSkipping Optuna, using default params from config")
    print(f"Params: {cfg.LGB_PARAMS}")

print("\nEvaluating (per-fold feature selection)...")
ev = evaluate(
    X_all,
    labels,
    cfg.LGB_PARAMS,
    n_folds=cfg.CV_FOLDS,
    seed=cfg.SEED,
    n_top=cfg.N_TOP_FEATURES,
)
print(
    f"\nTrain F1: {ev['train_f1']:.4f}  |  Val F1: {ev['val_f1']:.4f}  |  Gap: {ev['train_f1'] - ev['val_f1']:.4f}"
)
print(classification_report(labels, ev["preds"], target_names=list(cfg.CLASSES)))

cm = confusion_matrix(labels, ev["preds"])
print("Confusion Matrix:")
print("          Pred_H  Pred_O  Pred_R")
for i, name in enumerate(cfg.CLASSES):
    print(f"{name:>8}  {cm[i, 0]:>5}   {cm[i, 1]:>5}   {cm[i, 2]:>5}")

print("\nTraining final models (feature selection on full train)...")
models, sc, sel, vt = train_final(
    X_all,
    labels,
    cfg,
)

print("\nGenerating submission...")
test_names = [s["name"] for s in test_samples]
sub = predict(models, sc, X_test_all, test_names, cfg, sel=sel, vt=vt)

os.makedirs(cfg.OUT_DIR, exist_ok=True)
sub_path = os.path.join(cfg.OUT_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"\nSaved {sub_path} ({len(sub)} rows)")
print(sub["Category"].value_counts())
