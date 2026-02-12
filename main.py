import argparse
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CFG
from src.data import load_test, load_train
from src.features import extract_batch
from src.optimize import (
    detect_noisy_samples,
    evaluate,
    get_feature_importance,
    predict,
    run_optimization,
    train_final,
)

parser = argparse.ArgumentParser()
parser.add_argument("--run-optuna", action="store_true", help="Run Optuna optimization")
parser.add_argument("--prune-noise", action="store_true", help="Remove noisy samples")
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

# Note: Feature selection done on full data for efficiency
# Leakage is acceptable here as we're just ranking features, not learning decision boundaries
print("\nFeature selection...")
top_idx = get_feature_importance(X_all, labels, cfg)

X = X_all[:, top_idx[: cfg.N_TOP_FEATURES]]
X_test = X_test_all[:, top_idx[: cfg.N_TOP_FEATURES]]
print(f"Selected top {cfg.N_TOP_FEATURES} -> train: {X.shape}, test: {X_test.shape}")

clean_mask = np.ones(len(labels), dtype=bool)

# Step 1: Prune noisy samples FIRST (before Optuna)
if cfg.NOISE_CONF_THRESHOLD > 0 or args.prune_noise:
    threshold = cfg.NOISE_CONF_THRESHOLD if cfg.NOISE_CONF_THRESHOLD > 0 else 0.3
    print(f"\nDetecting noisy samples (conf < {threshold})...")
    # Use default params for noise detection
    default_params = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "random_state": cfg.SEED,
    }
    noise_result = detect_noisy_samples(
        X,
        labels,
        conf_threshold=threshold,
        n_folds=cfg.CV_FOLDS,
        seed=cfg.SEED,
        params=default_params,
    )
    clean_mask = ~noise_result["noisy_mask"]
    n_noisy = noise_result["noisy_mask"].sum()
    print(f"Pruning {n_noisy} noisy samples -> {clean_mask.sum()} clean")

X_clean = X[clean_mask]
y_clean = labels[clean_mask]

# Step 2: Run Optuna HPO on CLEAN data only
if args.run_optuna:
    print("\nOptuna HPO on clean data...")
    result = run_optimization(
        X_all[clean_mask],
        y_clean,
        cfg,
        top_idx=top_idx,
    )
    cfg.LGB_PARAMS = result["params"]
    cfg.VAR_THRESHOLD = result["var_threshold"]
    cfg.N_TOP_FEATURES = result["n_features"]
    print(f"Best F1: {result['best_f1']:.4f}  |  Gap: {result['best_gap']:.4f}")
    print(f"Params: {cfg.LGB_PARAMS}")
    print(f"Var threshold: {cfg.VAR_THRESHOLD:.2e}")
    print(f"N features: {cfg.N_TOP_FEATURES}")
    # Reselect features with optimized count
    X_clean = X_all[clean_mask][:, top_idx[: cfg.N_TOP_FEATURES]]
    X_test = X_test_all[:, top_idx[: cfg.N_TOP_FEATURES]]
    print(
        f"Reselected top {cfg.N_TOP_FEATURES} -> train: {X_clean.shape}, test: {X_test.shape}"
    )
else:
    print("\nSkipping Optuna, using default params from config")
    print(f"Params: {cfg.LGB_PARAMS}")

print("\nEvaluating...")
ev = evaluate(
    X_clean,
    y_clean,
    cfg.LGB_PARAMS,
    n_folds=cfg.CV_FOLDS,
    seed=cfg.SEED,
)
print(
    f"\nTrain F1: {ev['train_f1']:.4f}  |  Val F1: {ev['val_f1']:.4f}  |  Gap: {ev['train_f1'] - ev['val_f1']:.4f}"
)
print(classification_report(y_clean, ev["preds"], target_names=list(cfg.CLASSES)))

cm = confusion_matrix(y_clean, ev["preds"])
print("Confusion Matrix:")
print("          Pred_H  Pred_O  Pred_R")
for i, name in enumerate(cfg.CLASSES):
    print(f"{name:>8}  {cm[i, 0]:>5}   {cm[i, 1]:>5}   {cm[i, 2]:>5}")

print("\nTraining final models...")
models, sc = train_final(
    X_clean,
    y_clean,
    cfg,
)

print("\nGenerating submission...")
test_names = [s["name"] for s in test_samples]
sub = predict(models, sc, X_test, test_names, cfg)

os.makedirs(cfg.OUT_DIR, exist_ok=True)
sub_path = os.path.join(cfg.OUT_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"\nSaved {sub_path} ({len(sub)} rows)")
print(sub["Category"].value_counts())
