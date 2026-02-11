import os

import numpy as np
from sklearn.metrics import classification_report

from src.config import CFG
from src.data import load_test, load_train
from src.features import extract_batch
from src.optimize import (
    evaluate,
    generate_pseudo_labels,
    get_feature_importance,
    predict,
    run_optimization,
    train_final,
)

cfg = CFG()
cfg.OPTUNA_TRIALS = 40
cfg.CV_FOLDS = 5
cfg.N_TOP_FEATURES = 120

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

print("\nFeature selection...")
top_idx = get_feature_importance(X_all, labels, cfg)
X = X_all[:, top_idx[: cfg.N_TOP_FEATURES]]
X_test = X_test_all[:, top_idx[: cfg.N_TOP_FEATURES]]
print(f"Selected top {cfg.N_TOP_FEATURES} -> train: {X.shape}, test: {X_test.shape}")

print("\nPseudo-labeling...")
preds, conf, mask = generate_pseudo_labels(X, labels, X_test, cfg)
X_pseudo = X_test[mask]
y_pseudo = preds[mask]
dist = np.bincount(y_pseudo, minlength=3)
print(f"Pseudo-labels: {mask.sum()}/{len(X_test)} (threshold={cfg.PSEUDO_THRESHOLD})")
print(f"Distribution: H={dist[0]}, O={dist[1]}, R={dist[2]}")

RUN_OPTUNA = False

if RUN_OPTUNA:
    print("\nOptuna HPO...")
    result = run_optimization(
        X,
        labels,
        cfg,
        X_pseudo=X_pseudo,
        y_pseudo=y_pseudo,
        samples=samples,
        use_augmentation=True,
    )
    cfg.LGB_PARAMS = result["params"]
    cfg.HEALTH_WEIGHT = result["health_weight"]
    cfg.PSEUDO_WEIGHT = result.get("pseudo_weight", cfg.PSEUDO_WEIGHT)
    print(f"Best F1: {result['best_f1']:.4f}")
    print(f"Params: {cfg.LGB_PARAMS}")
    print(
        f"Health weight: {cfg.HEALTH_WEIGHT:.2f}, Pseudo weight: {cfg.PSEUDO_WEIGHT:.2f}"
    )
else:
    print("\nSkipping Optuna, using default params from config")
    print(f"Params: {cfg.LGB_PARAMS}")
    print(f"Health weight: {cfg.HEALTH_WEIGHT}, Pseudo weight: {cfg.PSEUDO_WEIGHT}")

print("\nEvaluating...")
ev = evaluate(
    X,
    labels,
    cfg.LGB_PARAMS,
    cfg.HEALTH_WEIGHT,
    cfg.CV_FOLDS,
    cfg.SEED,
    X_pseudo=X_pseudo,
    y_pseudo=y_pseudo,
    pseudo_weight=cfg.PSEUDO_WEIGHT,
    samples=samples,
    use_augmentation=True,
    aug_factor=2,
)
print(
    f"\nTrain F1: {ev['train_f1']:.4f}  |  Val F1: {ev['val_f1']:.4f}  |  Gap: {ev['train_f1'] - ev['val_f1']:.4f}"
)
print(classification_report(labels, ev["preds"], target_names=list(cfg.CLASSES)))

print("\nTraining final models...")
models, sc = train_final(
    X,
    labels,
    cfg,
    X_pseudo=X_pseudo,
    y_pseudo=y_pseudo,
    samples=samples,
    use_augmentation=True,
    aug_factor=3,
)

print("\nGenerating submission...")
test_names = [s["name"] for s in test_samples]
sub = predict(models, sc, X_test, test_names, cfg)

os.makedirs(cfg.OUT_DIR, exist_ok=True)
sub_path = os.path.join(cfg.OUT_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"\nSaved {sub_path} ({len(sub)} rows)")
print(sub["Category"].value_counts())
