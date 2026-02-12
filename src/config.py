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
            "n_estimators": 1200,
            "max_depth": 4,
            "learning_rate": 0.025,
            "subsample": 0.6,
            "colsample_bytree": 0.4,
            "min_child_samples": 30,
            "reg_alpha": 5.0,
            "reg_lambda": 5.0,
            "num_leaves": 8,
            "boosting_type": "gbdt",
            "min_split_gain": 0.05,
            "path_smooth": 1.0,
            "extra_trees": True,
            "class_weight": {0: 1.8, 1: 1.0, 2: 1.0},
        }
    )

    VAR_THRESHOLD: float = 1e-8
    ENSEMBLE_SEEDS: tuple = (42, 123, 456)

    BLANK_SAMPLES: frozenset = frozenset(
        {
            "Health_hyper_12",
            "Health_hyper_153",
            "Health_hyper_167",
            "Health_hyper_23",
            "Health_hyper_26",
            "Health_hyper_34",
            "Health_hyper_38",
            "Health_hyper_5",
            "Health_hyper_97",
            "Other_hyper_10",
            "Other_hyper_100",
            "Other_hyper_107",
            "Other_hyper_115",
            "Other_hyper_121",
            "Other_hyper_130",
            "Other_hyper_133",
            "Other_hyper_136",
            "Other_hyper_162",
            "Other_hyper_193",
            "Other_hyper_48",
            "Other_hyper_52",
            "Other_hyper_6",
            "Other_hyper_71",
        }
    )

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
