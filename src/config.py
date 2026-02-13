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

    N_TOP_FEATURES: int = 120
    OPTUNA_TRIALS: int = 40
    CV_FOLDS: int = 5
    SEED: int = 62

    LGB_PARAMS: dict = field(
        default_factory=lambda: {
            "n_estimators": 824,
            "max_depth": 5,
            "learning_rate": 0.01991217635443173,
            "subsample": 0.7231765814243538,
            "colsample_bytree": 0.41193352535022926,
            "min_child_samples": 40,
            "reg_alpha": 2.9366666590829906,
            "reg_lambda": 6.667228454493273,
            "num_leaves": 9,
            "boosting_type": "gbdt",
            "min_split_gain": 0.3475020292190679,
            "path_smooth": 0.006800104412862856,
            "extra_trees": True,
            "class_weight": {0: 1.302869968745372, 1: 1.0, 2: 1.0},
        }
    )

    VAR_THRESHOLD: float = 1.37e-09
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
