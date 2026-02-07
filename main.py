import argparse
import os

import numpy as np
import yaml
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--no-pseudo", action="store_true")
    args = parser.parse_args()

    cfg = CFG()
    cfg.OPTUNA_TRIALS = args.trials
    cfg.CV_FOLDS = args.folds
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    print("Loading data...")
    samples, labels = load_train(cfg)
    test_samples = load_test(cfg)
    print(f"  Train: {len(samples)}, Test: {len(test_samples)}")

    print("Extracting features...")
    X_all = extract_batch(samples)
    X_test_all = extract_batch(test_samples)
    print(f"  {X_all.shape[1]}d features")

    print("Feature selection...")
    top_idx = get_feature_importance(X_all, labels, cfg)
    X = X_all[:, top_idx[: cfg.N_TOP_FEATURES]]
    X_test = X_test_all[:, top_idx[: cfg.N_TOP_FEATURES]]
    print(f"  Selected top {cfg.N_TOP_FEATURES}")

    # Pseudo-labeling
    X_pseudo, y_pseudo = None, None
    if not args.no_pseudo:
        print(f"\nGenerating pseudo-labels (threshold={cfg.PSEUDO_THRESHOLD})...")
        preds, conf, mask = generate_pseudo_labels(X, labels, X_test, cfg)
        X_pseudo = X_test[mask]
        y_pseudo = preds[mask]
        dist = np.bincount(y_pseudo, minlength=3)
        print(f"  {mask.sum()}/{len(X_test)} confident pseudo-labels")
        print(f"  Distribution: H={dist[0]}, O={dist[1]}, R={dist[2]}")

    if not args.skip_optuna:
        print(f"\nOptuna HPO ({cfg.OPTUNA_TRIALS} trials, {cfg.CV_FOLDS}-fold)...")
        result = run_optimization(X, labels, cfg, X_pseudo, y_pseudo)
        cfg.LGB_PARAMS = result["params"]
        cfg.HEALTH_WEIGHT = result["health_weight"]
        cfg.PSEUDO_WEIGHT = result["pseudo_weight"]
        print(f"  Best F1: {result['best_f1']:.4f}")
        print(f"  Params: {cfg.LGB_PARAMS}")
        print(f"  Health weight: {cfg.HEALTH_WEIGHT:.2f}")
        print(f"  Pseudo weight: {cfg.PSEUDO_WEIGHT:.2f}")

        yaml.dump(
            {
                "lgb_params": cfg.LGB_PARAMS,
                "health_weight": cfg.HEALTH_WEIGHT,
                "pseudo_weight": cfg.PSEUDO_WEIGHT,
                "best_f1": float(result["best_f1"]),
                "n_top": cfg.N_TOP_FEATURES,
            },
            open(os.path.join(cfg.OUT_DIR, "best_params.yaml"), "w"),
        )

    print(f"\nFinal evaluation ({cfg.CV_FOLDS}-fold)...")
    ev = evaluate(
        X,
        labels,
        cfg.LGB_PARAMS,
        cfg.HEALTH_WEIGHT,
        cfg.CV_FOLDS,
        cfg.SEED,
        X_pseudo,
        y_pseudo,
        cfg.PSEUDO_WEIGHT,
    )
    print(
        f"  Train F1: {ev['train_f1']:.4f}  Val F1: {ev['val_f1']:.4f}  Gap: {ev['train_f1'] - ev['val_f1']:.4f}"
    )
    print(classification_report(labels, ev["preds"], target_names=list(cfg.CLASSES)))

    print("Training final model...")
    clf, sc = train_final(X, labels, cfg, X_pseudo, y_pseudo)

    print("Generating submission...")
    test_names = [s["name"] for s in test_samples]
    sub = predict(clf, sc, X_test, test_names, cfg)
    sub_path = os.path.join(cfg.OUT_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"  Saved to {sub_path} ({len(sub)} predictions)")
    print(f"  Distribution: {sub['Category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
