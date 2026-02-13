import os
from dataclasses import dataclass, field


@dataclass
class CFG:
    ROOT: str = "./data"
    TRAIN_DIR: str = "train"
    TEST_DIR: str = "test"
    OUT_DIR: str = "./outputs"

    CLASSES: tuple = ("Health", "Other", "Rust")
    N_CLASSES: int = 3

    N_TOP_FEATURES: int = 60
    OPTUNA_TRIALS: int = 40
    CV_FOLDS: int = 5
    SEED: int = 62

    LGB_PARAMS: dict = field(
        default_factory=lambda: {
            "n_estimators": 617,
            "max_depth": 5,
            "learning_rate": 0.06052076336782357,
            "subsample": 0.6698292354683619,
            "colsample_bytree": 0.3245341724635894,
            "min_child_samples": 32,
            "reg_alpha": 3.7944415835594603,
            "reg_lambda": 1.3637126525696954,
            "num_leaves": 14,
            "boosting_type": "gbdt",
            "min_split_gain": 0.2798089579584129,
            "path_smooth": 0.25651772044294585,
            "extra_trees": True,
            "class_weight": {0: 1.4948697962589994, 1: 1.0, 2: 1.0},
        }
    )

    VAR_THRESHOLD: float = 5.49e-10
    ENSEMBLE_SEEDS: tuple = (42, 123, 456)

    @property
    def class_map(self):
        return {c: i for i, c in enumerate(self.CLASSES)}

    @property
    def id_to_label(self):
        return {i: c for i, c in enumerate(self.CLASSES)}

    def train_path(self, modality):
        return os.path.join(self.ROOT, self.TRAIN_DIR, modality)

    def test_path(self, modality):
        return os.path.join(self.ROOT, self.TEST_DIR, modality)
