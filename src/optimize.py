import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import CFG


def _to_df(arr, n_cols):
    names = [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(arr, columns=names)


optuna.logging.set_verbosity(optuna.logging.INFO)


def mrmr_select(
    X: np.ndarray, y: np.ndarray, n_select: int, seed: int = 42
) -> np.ndarray:
    """mRMR feature selection (Peng, Long & Ding, IEEE TPAMI 2005).

    Greedy forward selection maximizing relevance (MI with target)
    minus mean redundancy (MI with already-selected features).
    Deterministic, fast, handles correlated spectral bands natively.
    Uses Pearson |r| as redundancy proxy (Ding & Peng 2005).
    """
    n_feats = X.shape[1]
    n_select = min(n_select, n_feats)

    relevance = mutual_info_classif(X, y, random_state=seed, n_neighbors=5)

    # Pre-compute full |correlation| matrix — O(n^2) but vectorized and fast
    abs_corr = np.abs(np.corrcoef(X, rowvar=False))
    np.nan_to_num(abs_corr, copy=False, nan=0.0)

    selected = []
    remaining = np.ones(n_feats, dtype=bool)
    # Running sum of correlations with selected features per candidate
    red_sum = np.zeros(n_feats, dtype=np.float64)

    for k in range(n_select):
        if k == 0:
            scores = relevance.copy()
        else:
            scores = relevance - red_sum / k
        scores[~remaining] = -np.inf
        best = int(np.argmax(scores))
        selected.append(best)
        remaining[best] = False
        # Update running redundancy sums for all remaining candidates
        red_sum += abs_corr[best]

    return np.array(selected, dtype=np.int64)


def _fit(clf, Xtr, ytr, Xval=None, yval=None, early_stop=50):
    if Xval is not None and yval is not None:
        clf.fit(
            Xtr,
            ytr,
            eval_set=[(Xval, yval)],
            callbacks=[
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    else:
        clf.fit(Xtr, ytr)
    return clf


def _build_pipeline(X, y, n_top, var_threshold, seed):
    """Shared mRMR -> VT -> scale -> fit pipeline used by evaluate and train_final."""
    sel = mrmr_select(X, y, n_select=n_top, seed=seed)
    X_sel = X[:, sel]
    vt = VarianceThreshold(threshold=var_threshold)
    X_sel = vt.fit_transform(X_sel)
    n = X_sel.shape[1]
    sc = StandardScaler()
    X_scaled = _to_df(sc.fit_transform(X_sel), n)
    return sel, vt, sc, X_scaled, n


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_folds: int = 5,
    seed: int = 42,
    n_top: int = 60,
    var_threshold: float = 0.0,
) -> dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=int)
    train_scores, val_scores, best_iters = [], [], []
    for fold_i, (tr, va) in enumerate(skf.split(X, y)):
        sel, vt, sc, Xtr, n = _build_pipeline(X[tr], y[tr], n_top, var_threshold, seed)
        Xva = _to_df(sc.transform(vt.transform(X[va][:, sel])), n)
        clf = LGBMClassifier(**params, verbose=-1, random_state=seed)
        _fit(clf, Xtr, y[tr], Xva, y[va], early_stop=50)
        actual_trees = getattr(clf, "best_iteration_", params.get("n_estimators", 0))
        best_iters.append(actual_trees)
        tr_f1 = f1_score(y[tr], clf.predict(Xtr), average="macro")
        preds[va] = clf.predict(Xva)
        va_f1 = f1_score(y[va], preds[va], average="macro")
        train_scores.append(tr_f1)
        val_scores.append(va_f1)
        print(
            f"  Fold {fold_i}: train_f1={tr_f1:.4f}  val_f1={va_f1:.4f}  gap={tr_f1 - va_f1:.4f}  trees={actual_trees}"
        )
    return {
        "val_f1": np.mean(val_scores),
        "train_f1": np.mean(train_scores),
        "preds": preds,
        "fold_train": train_scores,
        "fold_val": val_scores,
        "best_iters": best_iters,
    }


def run_optimization(
    X: np.ndarray,
    y: np.ndarray,
    cfg: CFG,
) -> dict:
    def objective(trial):
        n_features = trial.suggest_int("n_features", 40, 80, step=10)

        p = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.6),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 20),
            "boosting_type": "gbdt",
            "min_split_gain": trial.suggest_float("min_split_gain", 0.05, 0.5),
            "path_smooth": trial.suggest_float("path_smooth", 0.5, 10.0),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        }
        health_weight = trial.suggest_float("health_weight", 1.0, 2.5)
        p["class_weight"] = {0: health_weight, 1: 1.0, 2: 1.0}

        var_threshold = trial.suggest_float("var_threshold", 1e-10, 1e-4, log=True)

        skf = StratifiedKFold(
            n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.SEED
        )
        val_scores, train_scores = [], []
        for tr, va in skf.split(X, y):
            sel, vt_f, sc, Xtr, n = _build_pipeline(
                X[tr], y[tr], n_features, var_threshold, cfg.SEED
            )
            Xva = _to_df(sc.transform(vt_f.transform(X[va][:, sel])), n)
            clf = LGBMClassifier(**p, verbose=-1, random_state=cfg.SEED)
            _fit(clf, Xtr, y[tr], Xva, y[va], early_stop=50)
            train_scores.append(f1_score(y[tr], clf.predict(Xtr), average="macro"))
            val_scores.append(f1_score(y[va], clf.predict(Xva), average="macro"))
        mean_val = np.mean(val_scores)
        mean_train = np.mean(train_scores)
        mean_gap = mean_train - mean_val

        trial.set_user_attr("val_f1", mean_val)
        trial.set_user_attr("train_f1", mean_train)
        trial.set_user_attr("gap", mean_gap)

        # Progressive penalty: starts at gap=0.10, scales linearly
        gap_penalty = max(0, mean_gap - 0.10) * 1.0
        return mean_val - gap_penalty

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=cfg.SEED, n_startup_trials=min(15, cfg.OPTUNA_TRIALS // 3)
        ),
    )
    study.optimize(objective, n_trials=cfg.OPTUNA_TRIALS, show_progress_bar=True)

    bt = study.best_trial
    best = dict(bt.params)
    var_thresh = best.pop("var_threshold")
    health_weight = best.pop("health_weight", 1.5)
    n_features = best.pop("n_features", cfg.N_TOP_FEATURES)
    best["class_weight"] = {0: health_weight, 1: 1.0, 2: 1.0}
    return {
        "params": best,
        "var_threshold": var_thresh,
        "n_features": n_features,
        "best_f1": study.best_value,
        "best_gap": bt.user_attrs.get("gap", 0.0),
    }


def train_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: CFG,
    best_iters: list[int] = None,
):
    sel, vt, sc, Xtr, n = _build_pipeline(
        X_train, y_train, cfg.N_TOP_FEATURES, cfg.VAR_THRESHOLD, cfg.SEED
    )

    # Use median CV iteration count to avoid training beyond what CV validated
    if best_iters:
        cap = int(np.median(best_iters))
        final_params = {**cfg.LGB_PARAMS, "n_estimators": cap}
    else:
        final_params = cfg.LGB_PARAMS

    models = []
    for s in cfg.ENSEMBLE_SEEDS:
        clf = LGBMClassifier(**final_params, verbose=-1, random_state=s)
        clf.fit(Xtr, y_train)
        models.append(clf)
    print(
        f"Trained {len(models)} models (seeds: {cfg.ENSEMBLE_SEEDS}), "
        f"features: {n}, trees: {final_params.get('n_estimators')}"
    )
    return models, sc, sel, vt


def predict(
    models,
    sc,
    X_test: np.ndarray,
    test_names: list[str],
    cfg: CFG,
    sel: np.ndarray = None,
    vt: VarianceThreshold = None,
):
    if sel is not None:
        X_test = X_test[:, sel]
    if vt is not None:
        X_test = vt.transform(X_test)
    Xte = _to_df(sc.transform(X_test), X_test.shape[1])
    probs = np.zeros((len(X_test), cfg.N_CLASSES))
    for clf in models:
        probs += clf.predict_proba(Xte)
    probs /= len(models)
    preds = probs.argmax(1)
    return pd.DataFrame(
        {
            "Id": [n + ".tif" for n in test_names],
            "Category": [cfg.id_to_label[p] for p in preds],
        }
    )
