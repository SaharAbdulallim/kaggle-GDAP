import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.augment import create_augmented_batch
from src.config import CFG
from src.features import extract_batch


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


def _fit(clf, Xtr, ytr, w):
    clf.fit(Xtr, ytr, sample_weight=w)
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
    samples: list | None = None,
    use_augmentation: bool = True,
    aug_factor: int = 2,
    top_idx: np.ndarray | None = None,
) -> dict:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=int)
    train_scores, val_scores, best_iters = [], [], []
    n = X.shape[1]
    for fold_i, (tr, va) in enumerate(skf.split(X, y)):
        if use_augmentation and samples is not None:
            train_samples = [samples[i] for i in tr]
            aug_samples, aug_labels = create_augmented_batch(
                train_samples, y[tr], aug_factor=aug_factor
            )
            X_aug = extract_batch(aug_samples)
            if top_idx is not None:
                X_aug = X_aug[:, top_idx]
            if X_pseudo is not None and len(X_pseudo) > 0:
                X_fold = np.vstack([X_aug, X_pseudo])
                y_fold = np.concatenate([aug_labels, y_pseudo])
            else:
                X_fold, y_fold = X_aug, aug_labels
        else:
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
        _fit(clf, Xtr, y_fold, w)
        best_iters.append(params.get("n_estimators"))
        tr_f1 = f1_score(y_fold, clf.predict(Xtr), average="macro")
        preds[va] = clf.predict(Xva)
        va_f1 = f1_score(y[va], preds[va], average="macro")
        train_scores.append(tr_f1)
        val_scores.append(va_f1)
        print(
            f"  Fold {fold_i}: train_f1={tr_f1:.4f}  val_f1={va_f1:.4f}  gap={tr_f1 - va_f1:.4f}  trees={params.get('n_estimators')}"
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
    samples: list | None = None,
    use_augmentation: bool = True,
    top_idx: np.ndarray | None = None,
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
            "boosting_type": "dart",
            "drop_rate": trial.suggest_float("drop_rate", 0.05, 0.2),
            "skip_drop": trial.suggest_float("skip_drop", 0.3, 0.7),
            "max_drop": trial.suggest_int("max_drop", 20, 100),
        }
        hw = trial.suggest_float("health_weight", 1.0, 3.0)
        pw = (
            trial.suggest_float("pseudo_weight", 0.1, 1.0)
            if X_pseudo is not None
            else 0.5
        )
        var_threshold = trial.suggest_float("var_threshold", 1e-10, 1e-4, log=True)

        selector = VarianceThreshold(threshold=var_threshold)
        X_filtered = selector.fit_transform(X)
        n = X_filtered.shape[1]
        skf = StratifiedKFold(
            n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.SEED
        )
        val_scores, train_scores = [], []
        for tr, va in skf.split(X_filtered, y):
            if use_augmentation and samples is not None:
                train_samples = [samples[i] for i in tr]
                aug_samples, aug_labels = create_augmented_batch(
                    train_samples, y[tr], aug_factor=1
                )
                X_aug = extract_batch(aug_samples)
                if top_idx is not None:
                    X_aug = X_aug[:, top_idx]
                X_aug_filtered = selector.transform(X_aug)
                if X_pseudo is not None and len(X_pseudo) > 0:
                    X_pseudo_filtered = selector.transform(X_pseudo)
                    X_fold = np.vstack([X_aug_filtered, X_pseudo_filtered])
                    y_fold = np.concatenate([aug_labels, y_pseudo])
                else:
                    X_fold, y_fold = X_aug_filtered, aug_labels
            else:
                if X_pseudo is not None and len(X_pseudo) > 0:
                    X_pseudo_filtered = selector.transform(X_pseudo)
                    X_fold = np.vstack([X_filtered[tr], X_pseudo_filtered])
                    y_fold = np.concatenate([y[tr], y_pseudo])
                else:
                    X_fold, y_fold = X_filtered[tr], y[tr]
            sc = StandardScaler()
            Xtr = _to_df(sc.fit_transform(X_fold), n)
            Xva = _to_df(sc.transform(X_filtered[va]), n)
            w = np.ones(len(y_fold))
            w[y_fold == 0] *= hw
            if X_pseudo is not None and len(X_pseudo) > 0:
                w[len(y[tr]) :] *= pw
            clf = LGBMClassifier(**p, verbose=-1, random_state=cfg.SEED)
            _fit(clf, Xtr, y_fold, w)
            train_scores.append(f1_score(y_fold, clf.predict(Xtr), average="macro"))
            val_scores.append(f1_score(y[va], clf.predict(Xva), average="macro"))
        mean_val = np.mean(val_scores)
        mean_gap = np.mean(train_scores) - mean_val
        trial.set_user_attr("val_f1", mean_val)
        trial.set_user_attr("gap", mean_gap)
        return mean_val

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=cfg.SEED, n_startup_trials=min(15, cfg.OPTUNA_TRIALS // 3)
        ),
    )
    study.optimize(objective, n_trials=cfg.OPTUNA_TRIALS)

    bt = study.best_trial
    best = dict(bt.params)
    hw = best.pop("health_weight")
    pw = best.pop("pseudo_weight", cfg.PSEUDO_WEIGHT)
    var_thresh = best.pop("var_threshold")
    return {
        "params": best,
        "health_weight": hw,
        "pseudo_weight": pw,
        "var_threshold": var_thresh,
        "best_f1": study.best_value,
        "best_gap": bt.user_attrs.get("gap", 0.0),
    }


def train_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: CFG,
    X_pseudo: np.ndarray | None = None,
    y_pseudo: np.ndarray | None = None,
    samples: list | None = None,
    use_augmentation: bool = True,
    aug_factor: int = 3,
    top_idx: np.ndarray | None = None,
):
    if use_augmentation and samples is not None:
        aug_samples, aug_labels = create_augmented_batch(
            samples, y_train, aug_factor=aug_factor
        )
        X_aug = extract_batch(aug_samples)
        if top_idx is not None:
            X_aug = X_aug[:, top_idx]
        if X_pseudo is not None and len(X_pseudo) > 0:
            X_all = np.vstack([X_aug, X_pseudo])
            y_all = np.concatenate([aug_labels, y_pseudo])
        else:
            X_all, y_all = X_aug, aug_labels
    else:
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

    models = []
    for s in cfg.PSEUDO_SEEDS:
        clf = LGBMClassifier(**cfg.LGB_PARAMS, verbose=-1, random_state=s)
        clf.fit(Xtr, y_all, sample_weight=w)
        models.append(clf)
    print(f"Trained {len(models)} models (seeds: {cfg.PSEUDO_SEEDS})")
    return models, sc


def predict(models, sc, X_test: np.ndarray, test_names: list[str], cfg: CFG):
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
