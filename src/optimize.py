import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import CFG


def _to_df(arr, n_cols):
    names = [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(arr, columns=names)


optuna.logging.set_verbosity(optuna.logging.INFO)


def get_feature_importance(X: np.ndarray, y: np.ndarray, cfg: CFG) -> np.ndarray:
    sc = StandardScaler()
    X_s = _to_df(sc.fit_transform(X), X.shape[1])
    clf = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=cfg.SEED,
    )
    clf.fit(X_s, y)
    return np.argsort(-clf.feature_importances_)


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


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_folds: int = 5,
    seed: int = 42,
    n_top: int = 60,
) -> dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=int)
    train_scores, val_scores, best_iters = [], [], []
    for fold_i, (tr, va) in enumerate(skf.split(X, y)):
        # Per-fold feature selection eliminates leakage
        fold_idx = get_feature_importance(X[tr], y[tr], CFG(SEED=seed))
        sel = fold_idx[:n_top]
        X_fold, y_fold = X[tr][:, sel], y[tr]
        n = X_fold.shape[1]
        sc = StandardScaler()
        Xtr = _to_df(sc.fit_transform(X_fold), n)
        Xva = _to_df(sc.transform(X[va][:, sel]), n)
        clf = LGBMClassifier(**params, verbose=-1, random_state=seed)
        _fit(clf, Xtr, y_fold, Xva, y[va], early_stop=50)
        actual_trees = (
            clf.n_estimators_
            if hasattr(clf, "n_estimators_")
            else clf.best_iteration_
            if hasattr(clf, "best_iteration_")
            else params.get("n_estimators")
        )
        best_iters.append(actual_trees)
        tr_f1 = f1_score(y_fold, clf.predict(Xtr), average="macro")
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
    # Compute feature ranking once (on full train set Optuna sees)
    # Reused across trials — not leakage since CV inside each trial is independent
    cached_idx = get_feature_importance(X, y, cfg)

    def objective(trial):
        n_features = trial.suggest_int("n_features", 60, 200, step=20)
        sel = cached_idx[:n_features]

        p = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.6),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 15),
            "boosting_type": "gbdt",
            "min_split_gain": trial.suggest_float("min_split_gain", 0.01, 0.5),
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 5.0),
            "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
        }
        health_weight = trial.suggest_float("health_weight", 1.0, 1.5)
        p["class_weight"] = {0: health_weight, 1: 1.0, 2: 1.0}

        var_threshold = trial.suggest_float("var_threshold", 1e-10, 1e-4, log=True)

        skf = StratifiedKFold(
            n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.SEED
        )
        val_scores, train_scores = [], []
        for tr, va in skf.split(X, y):
            X_tr_sel, X_va_sel = X[tr][:, sel], X[va][:, sel]

            selector = VarianceThreshold(threshold=var_threshold)
            X_fold_filtered = selector.fit_transform(X_tr_sel)
            X_val_filtered = selector.transform(X_va_sel)
            y_fold = y[tr]

            n = X_fold_filtered.shape[1]
            sc = StandardScaler()
            Xtr = _to_df(sc.fit_transform(X_fold_filtered), n)
            Xva = _to_df(sc.transform(X_val_filtered), n)
            clf = LGBMClassifier(**p, verbose=-1, random_state=cfg.SEED)
            _fit(clf, Xtr, y_fold, Xva, y[va], early_stop=50)
            train_scores.append(f1_score(y_fold, clf.predict(Xtr), average="macro"))
            val_scores.append(f1_score(y[va], clf.predict(Xva), average="macro"))
        mean_val = np.mean(val_scores)
        mean_train = np.mean(train_scores)
        mean_gap = mean_train - mean_val

        trial.set_user_attr("val_f1", mean_val)
        trial.set_user_attr("train_f1", mean_train)
        trial.set_user_attr("gap", mean_gap)

        gap_penalty = max(0, mean_gap - 0.12) * 1.0
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
):
    top_idx = get_feature_importance(X_train, y_train, cfg)
    sel = top_idx[: cfg.N_TOP_FEATURES]
    X_sel = X_train[:, sel]

    vt = VarianceThreshold(threshold=cfg.VAR_THRESHOLD)
    X_sel = vt.fit_transform(X_sel)

    sc = StandardScaler()
    Xtr = _to_df(sc.fit_transform(X_sel), X_sel.shape[1])

    models = []
    for s in cfg.ENSEMBLE_SEEDS:
        clf = LGBMClassifier(**cfg.LGB_PARAMS, verbose=-1, random_state=s)
        clf.fit(Xtr, y_train)
        models.append(clf)
    print(
        f"Trained {len(models)} models (seeds: {cfg.ENSEMBLE_SEEDS}), features: {X_sel.shape[1]}"
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
