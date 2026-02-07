import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import CFG

EARLY_STOP_ROUNDS = 50


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


def _fit_with_early_stop(clf, Xtr, ytr, Xva, yva, w):
    clf.fit(
        Xtr,
        ytr,
        sample_weight=w,
        eval_set=[(Xva, yva)],
        callbacks=[
            early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            log_evaluation(-1),
        ],
    )
    return clf


def generate_pseudo_labels(X_train, y_train, X_test, cfg: CFG):
    sc = StandardScaler()
    n = X_train.shape[1]
    Xtr = _to_df(sc.fit_transform(X_train), n)
    Xte = _to_df(sc.transform(X_test), n)
    probs = np.zeros((len(X_test), cfg.N_CLASSES))
    for seed in cfg.PSEUDO_SEEDS:
        clf = LGBMClassifier(**cfg.LGB_PARAMS, verbose=-1, random_state=seed)
        w = np.ones(len(y_train))
        w[y_train == 0] = cfg.HEALTH_WEIGHT
        clf.fit(Xtr, y_train, sample_weight=w)
        probs += clf.predict_proba(Xte)
    probs /= len(cfg.PSEUDO_SEEDS)
    preds = probs.argmax(1)
    conf = probs.max(1)
    mask = conf >= cfg.PSEUDO_THRESHOLD
    return preds, conf, mask


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    health_weight: float,
    n_folds: int = 5,
    seed: int = 42,
    X_pseudo: np.ndarray | None = None,
    y_pseudo: np.ndarray | None = None,
    pseudo_weight: float = 0.5,
) -> dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=int)
    train_scores, val_scores, best_iters = [], [], []
    n = X.shape[1]
    for fold_i, (tr, va) in enumerate(skf.split(X, y)):
        if X_pseudo is not None and len(X_pseudo) > 0:
            X_fold = np.vstack([X[tr], X_pseudo])
            y_fold = np.concatenate([y[tr], y_pseudo])
        else:
            X_fold, y_fold = X[tr], y[tr]
        sc = StandardScaler()
        Xtr = _to_df(sc.fit_transform(X_fold), n)
        Xva = _to_df(sc.transform(X[va]), n)
        w = np.ones(len(y_fold))
        w[y_fold == 0] = health_weight
        if X_pseudo is not None and len(X_pseudo) > 0:
            w[len(y[tr]) :] *= pseudo_weight
        clf = LGBMClassifier(**params, verbose=-1, random_state=seed)
        _fit_with_early_stop(clf, Xtr, y_fold, Xva, y[va], w)
        best_iters.append(clf.best_iteration_)
        tr_f1 = f1_score(y_fold, clf.predict(Xtr), average="macro")
        preds[va] = clf.predict(Xva)
        va_f1 = f1_score(y[va], preds[va], average="macro")
        train_scores.append(tr_f1)
        val_scores.append(va_f1)
        print(
            f"  Fold {fold_i}: train_f1={tr_f1:.4f}  val_f1={va_f1:.4f}  gap={tr_f1 - va_f1:.4f}  trees={clf.best_iteration_}"
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
    X_pseudo: np.ndarray | None = None,
    y_pseudo: np.ndarray | None = None,
) -> dict:
    def objective(trial):
        p = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.06, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.4, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 20.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 50),
        }
        hw = trial.suggest_float("health_weight", 1.0, 3.0)
        pw = (
            trial.suggest_float("pseudo_weight", 0.1, 1.0)
            if X_pseudo is not None
            else 0.5
        )
        n = X.shape[1]
        skf = StratifiedKFold(
            n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.SEED
        )
        val_scores = []
        for tr, va in skf.split(X, y):
            if X_pseudo is not None and len(X_pseudo) > 0:
                X_fold = np.vstack([X[tr], X_pseudo])
                y_fold = np.concatenate([y[tr], y_pseudo])
            else:
                X_fold, y_fold = X[tr], y[tr]
            sc = StandardScaler()
            Xtr = _to_df(sc.fit_transform(X_fold), n)
            Xva = _to_df(sc.transform(X[va]), n)
            w = np.ones(len(y_fold))
            w[y_fold == 0] *= hw
            if X_pseudo is not None and len(X_pseudo) > 0:
                w[len(y[tr]) :] *= pw
            clf = LGBMClassifier(**p, verbose=-1, random_state=cfg.SEED)
            _fit_with_early_stop(clf, Xtr, y_fold, Xva, y[va], w)
            val_scores.append(f1_score(y[va], clf.predict(Xva), average="macro"))
        return np.mean(val_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=cfg.SEED, n_startup_trials=min(15, cfg.OPTUNA_TRIALS // 3)
        ),
    )
    study.optimize(objective, n_trials=cfg.OPTUNA_TRIALS)

    best = study.best_params
    hw = best.pop("health_weight")
    pw = best.pop("pseudo_weight", cfg.PSEUDO_WEIGHT)
    return {
        "params": best,
        "health_weight": hw,
        "pseudo_weight": pw,
        "best_f1": study.best_value,
    }


def train_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: CFG,
    X_pseudo: np.ndarray | None = None,
    y_pseudo: np.ndarray | None = None,
):
    if X_pseudo is not None and len(X_pseudo) > 0:
        X_all = np.vstack([X_train, X_pseudo])
        y_all = np.concatenate([y_train, y_pseudo])
    else:
        X_all, y_all = X_train, y_train

    sc = StandardScaler()
    Xtr = _to_df(sc.fit_transform(X_all), X_train.shape[1])
    w = np.ones(len(y_all))
    w[y_all == 0] = cfg.HEALTH_WEIGHT
    if X_pseudo is not None and len(X_pseudo) > 0:
        w[len(y_train) :] *= cfg.PSEUDO_WEIGHT
    clf = LGBMClassifier(**cfg.LGB_PARAMS, verbose=-1, random_state=cfg.SEED)
    clf.fit(Xtr, y_all, sample_weight=w)
    return clf, sc


def predict(clf, sc, X_test: np.ndarray, test_names: list[str], cfg: CFG):
    Xte = _to_df(sc.transform(X_test), X_test.shape[1])
    preds = clf.predict(Xte)
    return pd.DataFrame(
        {
            "Id": [n + ".tif" for n in test_names],
            "Category": [cfg.id_to_label[p] for p in preds],
        }
    )
