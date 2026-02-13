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
    OPTUNA_TRIALS: int = 60
    CV_FOLDS: int = 5
    SEED: int = 62

    LGB_PARAMS: dict = field(
        default_factory=lambda: {
            "n_estimators": 800,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.4,
            "min_child_samples": 50,
            "reg_alpha": 10.0,
            "reg_lambda": 10.0,
            "num_leaves": 15,
            "boosting_type": "gbdt",
            "min_split_gain": 0.2,
            "path_smooth": 3.0,
            "extra_trees": False,
            "class_weight": {0: 1.5, 1: 1.0, 2: 1.0},
        }
    )

    VAR_THRESHOLD: float = 1e-08
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
